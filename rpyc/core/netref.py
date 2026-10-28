"""*NetRef*: a transparent *network reference*. This module contains quite a lot
of *magic*, so beware.
"""
import sys
import types
import weakref
from threading import RLock
from rpyc.lib import get_methods, get_id_pack
from rpyc.lib.compat import pickle, maxint
from rpyc.core import consts


builtin_id_pack_cache = {}  # name_pack -> id_pack
builtin_classes_cache = {}  # id_pack -> class


class NetrefScope(object):
    """Identity token grouping connections that belong to one logical RPyC node.

    Connections constructed with the *same* scope instance share their netref
    registry: because the ``id_pack`` of a remote object is preserved across
    relay (RPyC-over-RPyC) hops, the same source object arriving over several
    independent connections is represented by a single, multi-routed netref
    instead of one netref per connection.

    Give one scope per logical node (e.g. a client process passes the same
    scope to both of its relay connections). In a real deployment every node is
    a separate process, so scopes are only needed when several nodes are
    emulated inside one process; leaving the connection configuration unset
    gives every connection a private scope (the historical behavior).

    Instances compare and hash by identity and are weakly referenceable
    because they are used as keys of a :class:`weakref.WeakKeyDictionary`.

    Note: ``id()`` is only unique within a single address space, so a scope
    may only group connections whose peers genuinely share one address space
    (the RPyC relay use case these registries serve).
    """
    __slots__ = ("__weakref__",)

    def __repr__(self):
        return f"<netref-scope @{id(self):x}>"


#: Registries of netrefs keyed by the source-object ``id_pack``, partitioned
#: by :class:`NetrefScope`. Maps ``scope -> (id_pack -> netref)``; both levels
#: are weak so neither scopes nor proxies are kept alive by the registry.
_shared_proxy_cache = weakref.WeakKeyDictionary()
_shared_proxy_lock = RLock()


def _shared_scope_cache(scope, create=False):
    cache = _shared_proxy_cache.get(scope)
    if cache is None and create:
        cache = weakref.WeakValueDictionary()
        _shared_proxy_cache[scope] = cache
    return cache


# If these can be accessed, numpy will try to load the array from local memory,
# resulting in exceptions and/or segfaults, see #236:
DELETED_ATTRS = frozenset([
    '__array_struct__', '__array_interface__',
])

"""the set of attributes that are local to the netref object"""
LOCAL_ATTRS = frozenset([
    '____conn__', '____id_pack__', '____refcount__', '____routes__', '____rlock__',
    '____scope__',
    '__class__', '__cmp__', '__del__', '__delattr__',
    '__dir__', '__doc__', '__getattr__', '__getattribute__', '__hash__', '__instancecheck__',
    '__init__', '__metaclass__', '__module__', '__new__', '__reduce__',
    '__reduce_ex__', '__repr__', '__setattr__', '__slots__', '__str__',
    '__weakref__', '__dict__', '__methods__', '__exit__',
    '__eq__', '__ne__', '__lt__', '__gt__', '__le__', '__ge__',
]) | DELETED_ATTRS

"""a list of types considered built-in (shared between connections)
this is needed because iterating the members of the builtins module is not enough,
some types (e.g NoneType) are not members of the builtins module.
TODO: this list is not complete.
"""
_builtin_types = [
    type, object, bool, complex, dict, float, int, list, slice, str, tuple, set,
    frozenset, BaseException, Exception, type(None), types.BuiltinFunctionType, types.GeneratorType,
    types.MethodType, types.CodeType, types.FrameType, types.TracebackType,
    types.ModuleType, types.FunctionType, types.MappingProxyType,

    type(int.__add__),      # wrapper_descriptor
    type((1).__add__),      # method-wrapper
    type(iter([])),         # listiterator
    type(iter(())),         # tupleiterator
    type(iter(set())),      # setiterator
    bytes, bytearray, type(iter(range(10))), memoryview
]
_normalized_builtin_types = {}


# ------------------------------------------------------------------------------
# multi-path routing
# ------------------------------------------------------------------------------
# A netref may be reachable through several independent connections (typically
# RPyC-over-RPyC relay chains leading to the same source object). Each entry in
# the proxy's ``____routes__`` map is ``connection -> local refcount``, where
# the refcount equals the number of ``LABEL_REMOTE_REF`` occurrences received
# over that connection -- each one corresponding to a remote-side incref.

def _get_routes(proxy):
    return object.__getattribute__(proxy, "____routes__")


def _get_route_lock(proxy):
    return object.__getattribute__(proxy, "____rlock__")


def _route_alive(conn):
    """Whether a request may still travel over ``conn``.

    ``Connection.close`` sets its ``closed`` flag *before* invoking the
    ``before_closed`` hook and sending ``HANDLE_CLOSE`` so a final request can
    ride the still-open channel; route liveness therefore follows the
    underlying channel/stream, which only closes during ``_cleanup``.
    """
    if conn is None:
        return False
    if not conn.closed:
        return True
    channel = getattr(conn, "_channel", None)
    return channel is not None and not channel.closed


def get_live_conn(proxy):
    """Returns a connection over which the proxied object is reachable.

    The most recently used connection is preferred while it is alive;
    otherwise the first open route is chosen and cached in ``____conn__``.
    Returns ``None`` when every route has been closed.
    """
    with _get_route_lock(proxy):
        conn = object.__getattribute__(proxy, "____conn__")
        if _route_alive(conn):
            return conn
        for candidate in _get_routes(proxy):
            if _route_alive(candidate):
                object.__setattr__(proxy, "____conn__", candidate)
                return candidate
        object.__setattr__(proxy, "____conn__", None)
        return None


def add_route(proxy, conn):
    """Registers ``conn`` as an additional path to the proxied object and
    accounts for one more remote-side reference held over that connection."""
    with _get_route_lock(proxy):
        routes = _get_routes(proxy)
        routes[conn] = routes.get(conn, 0) + 1
        object.__setattr__(proxy, "____refcount__",
                           object.__getattribute__(proxy, "____refcount__") + 1)
        if not _route_alive(object.__getattribute__(proxy, "____conn__")):
            object.__setattr__(proxy, "____conn__", conn)


def retire_connection(proxy, conn):
    """Removes a (closed) connection from the set of usable paths.

    When no route remains the proxy is also dropped from the shared registry,
    so that a later re-delivery of the same ``id_pack`` creates a fresh proxy
    instead of resurrecting a dead one. Returns ``True`` if routes remain.
    """
    with _get_route_lock(proxy):
        routes = _get_routes(proxy)
        routes.pop(conn, None)
        current = object.__getattribute__(proxy, "____conn__")
        if current is conn or not _route_alive(current):
            replacement = next((c for c in routes if _route_alive(c)), None)
            object.__setattr__(proxy, "____conn__", replacement)
        id_pack = object.__getattribute__(proxy, "____id_pack__")
        scope = object.__getattribute__(proxy, "____scope__")
        empty = not routes
    if empty and scope is not None:
        with _shared_proxy_lock:
            cache = _shared_scope_cache(scope)
            if cache is not None and cache.get(id_pack) is proxy:
                cache.pop(id_pack, None)
    return not empty


def discard_routes(proxy):
    """Detaches every route without sending ``HANDLE_DEL``.

    Used for netrefs that were constructed but lost the registration race to an
    equivalent proxy: their single reference is accounted for on the winning
    proxy instead, so no release message must be sent when they are collected.
    """
    with _get_route_lock(proxy):
        _get_routes(proxy).clear()
        object.__setattr__(proxy, "____conn__", None)


def has_route(proxy, conn):
    """Returns whether the peer of ``conn`` already holds the proxied object,
    i.e. whether it can be shipped back as a ``LABEL_LOCAL_REF``."""
    with _get_route_lock(proxy):
        return conn in _get_routes(proxy)


def shared_lookup(scope, id_pack):
    """Returns the shared proxy for ``id_pack`` within ``scope`` if one with a
    live route exists, ``None`` otherwise."""
    with _shared_proxy_lock:
        cache = _shared_scope_cache(scope)
        proxy = cache.get(id_pack) if cache is not None else None
        if proxy is not None and get_live_conn(proxy) is None:
            proxy = None
        return proxy


def shared_register(scope, id_pack, proxy):
    """Publishes ``proxy`` in the shared registry of ``scope`` unless another
    live proxy is already registered for ``id_pack``; returns the proxy to use
    (the existing winner or ``proxy`` itself)."""
    with _shared_proxy_lock:
        cache = _shared_scope_cache(scope, create=True)
        existing = cache.get(id_pack)
        if existing is not None and existing is not proxy and get_live_conn(existing) is not None:
            return existing
        cache[id_pack] = proxy
        return proxy


def syncreq(proxy, handler, *args):
    """Performs a synchronous request on the given proxy object.
    Not intended to be invoked directly.

    When the proxy is reachable through several connections, any live route
    may be used; if a route breaks while serving the request, the call is
    retried over the next live route (a request may thus be re-executed).

    :param proxy: the proxy on which to issue the request
    :param handler: the request handler (one of the ``HANDLE_XXX`` members of
                    ``rpyc.protocol.consts``)
    :param args: arguments to the handler

    :raises: any exception raised by the operation will be raised
    :returns: the result of the operation
    """
    last_error = None
    attempted = set()
    while True:
        conn = get_live_conn(proxy)
        if conn is None or id(conn) in attempted:
            if last_error is not None:
                raise last_error
            raise EOFError("the remote object is unreachable: all routing "
                           "connections have been closed")
        attempted.add(id(conn))
        try:
            return conn.sync_request(handler, proxy, *args)
        except (EOFError, OSError) as ex:
            # connection-level failure: retire the broken route and fail over
            last_error = ex
            retire_connection(proxy, conn)


def asyncreq(proxy, handler, *args):
    """Performs an asynchronous request on the given proxy object.
    Not intended to be invoked directly.

    The request is issued on any live route; unlike :func:`syncreq` there is
    no automatic fail-over since the caller is not waiting on the result.

    :param proxy: the proxy on which to issue the request
    :param handler: the request handler (one of the ``HANDLE_XXX`` members of
                    ``rpyc.protocol.consts``)
    :param args: arguments to the handler

    :returns: an :class:`~rpyc.core.async_.AsyncResult` representing
              the operation
    """
    conn = get_live_conn(proxy)
    if conn is None:
        raise EOFError("the remote object is unreachable: all routing "
                       "connections have been closed")
    return conn.async_request(handler, proxy, *args)


class NetrefMetaclass(type):
    """A *metaclass* used to customize the ``__repr__`` of ``netref`` classes.
    It is quite useless, but it makes debugging and interactive programming
    easier"""
    __slots__ = ()

    def __repr__(self):
        if self.__module__:
            return f"<netref class '{self.__module__}.{self.__name__}'>"
        else:
            return f"<netref class '{self.__name__}'>"


class BaseNetref(object, metaclass=NetrefMetaclass):
    """The base netref class, from which all netref classes derive. Some netref
    classes are "pre-generated" and cached upon importing this module (those
    defined in the :data:`_builtin_types`), and they are shared between all
    connections.

    The rest of the netref classes are created by :meth:`rpyc.core.protocol.Connection._unbox`,
    and are private to the connection.

    Do not use this class directly; use :func:`class_factory` instead.

    :param conn: the :class:`rpyc.core.protocol.Connection` instance
    :param id_pack: id tuple for an object ~ (name_pack, remote-class-id, remote-instance-id)
        (cont.) name_pack := __module__.__name__ (hits or misses on builtin cache and sys.module)
                remote-class-id := id of object class (hits or misses on netref classes cache and instance checks)
                remote-instance-id := id object instance (hits or misses on proxy cache)
        id_pack is usually created by rpyc.lib.get_id_pack
    """
    __slots__ = ["____conn__", "____id_pack__", "__weakref__", "____refcount__",
                 "____routes__", "____rlock__", "____scope__"]

    def __init__(self, conn, id_pack):
        object.__setattr__(self, "____id_pack__", id_pack)
        object.__setattr__(self, "____rlock__", RLock())
        object.__setattr__(self, "____routes__", {} if conn is None else {conn: 1})
        object.__setattr__(self, "____conn__", conn)
        object.__setattr__(self, "____refcount__", 1)
        object.__setattr__(self, "____scope__",
                           None if conn is None else getattr(conn, "_netref_scope", None))

    def __del__(self):
        # Release the references held over every still-open route.
        # Each connection owns an independent remote refcount; connections
        # already closed have dropped theirs in ``Connection._cleanup``.
        try:
            with _get_route_lock(self):
                routes = dict(_get_routes(self))
            if not routes:
                return
            for conn, count in routes.items():
                if not _route_alive(conn):
                    continue
                try:
                    # boxing still maps this netref to LABEL_LOCAL_REF while
                    # the routes are intact, so the id pack is what is sent
                    conn.async_request(consts.HANDLE_DEL, self, count)
                except Exception:
                    # raised in a destructor, most likely on program
                    # termination, when the connection might have been
                    # closed; the other side clears references on disconnect
                    pass
            with _get_route_lock(self):
                _get_routes(self).clear()
                object.__setattr__(self, "____conn__", None)
        except Exception:
            # it is safe to ignore all exceptions here
            pass

    def __getattribute__(self, name):
        if name in LOCAL_ATTRS:
            if name == "__class__":
                cls = object.__getattribute__(self, "__class__")
                if cls is None:
                    cls = self.__getattr__("__class__")
                return cls
            elif name == "__doc__":
                return self.__getattr__("__doc__")
            elif name in DELETED_ATTRS:
                raise AttributeError()
            else:
                return object.__getattribute__(self, name)
        elif name == "__call__":                          # IronPython issue #10
            return object.__getattribute__(self, "__call__")
        elif name == "__array__":
            return object.__getattribute__(self, "__array__")
        else:
            return syncreq(self, consts.HANDLE_GETATTR, name)

    def __getattr__(self, name):
        if name in DELETED_ATTRS:
            raise AttributeError()
        return syncreq(self, consts.HANDLE_GETATTR, name)

    def __delattr__(self, name):
        if name in LOCAL_ATTRS:
            object.__delattr__(self, name)
        else:
            syncreq(self, consts.HANDLE_DELATTR, name)

    def __setattr__(self, name, value):
        if name in LOCAL_ATTRS:
            object.__setattr__(self, name, value)
        else:
            syncreq(self, consts.HANDLE_SETATTR, name, value)

    def __dir__(self):
        return list(syncreq(self, consts.HANDLE_DIR))

    # support for metaclasses
    def __hash__(self):
        return syncreq(self, consts.HANDLE_HASH)

    def __cmp__(self, other):
        return syncreq(self, consts.HANDLE_CMP, other, '__cmp__')

    def __eq__(self, other):
        return syncreq(self, consts.HANDLE_CMP, other, '__eq__')

    def __ne__(self, other):
        return syncreq(self, consts.HANDLE_CMP, other, '__ne__')

    def __lt__(self, other):
        return syncreq(self, consts.HANDLE_CMP, other, '__lt__')

    def __gt__(self, other):
        return syncreq(self, consts.HANDLE_CMP, other, '__gt__')

    def __le__(self, other):
        return syncreq(self, consts.HANDLE_CMP, other, '__le__')

    def __ge__(self, other):
        return syncreq(self, consts.HANDLE_CMP, other, '__ge__')

    def __repr__(self):
        return syncreq(self, consts.HANDLE_REPR)

    def __str__(self):
        return syncreq(self, consts.HANDLE_STR)

    def __exit__(self, exc, typ, tb):
        return syncreq(self, consts.HANDLE_CTXEXIT, exc)  # can't pass type nor traceback

    def __reduce_ex__(self, proto):
        # support for pickling netrefs
        return pickle.loads, (syncreq(self, consts.HANDLE_PICKLE, proto),)

    def __instancecheck__(self, other):
        # support for checking cached instances across connections
        if isinstance(other, BaseNetref):
            if self.____id_pack__[2] != 0:
                raise TypeError("isinstance() arg 2 must be a class, type, or tuple of classes and types")
            elif self.____id_pack__[1] == other.____id_pack__[1]:
                if other.____id_pack__[2] == 0:
                    return False
                elif other.____id_pack__[2] != 0:
                    return True
            else:
                # seems dubious if each netref proxies to a different address spaces
                return syncreq(self, consts.HANDLE_INSTANCECHECK, other.____id_pack__)
        else:
            if self.____id_pack__[2] == 0:
                # outside the context of `__instancecheck__`, `__class__` is expected to be type(self)
                # within the context of `__instancecheck__`, `other` should be compared to the proxied class
                return isinstance(other, type(self).__dict__['__class__'].instance)
            else:
                raise TypeError("isinstance() arg 2 must be a class, type, or tuple of classes and types")


def _make_method(name, doc):
    """creates a method with the given name and docstring that invokes
    :func:`syncreq` on its `self` argument"""

    slicers = {"__getslice__": "__getitem__", "__delslice__": "__delitem__", "__setslice__": "__setitem__"}

    name = str(name)                                      # IronPython issue #10
    if name == "__call__":
        def __call__(_self, *args, **kwargs):
            kwargs = tuple(kwargs.items())
            return syncreq(_self, consts.HANDLE_CALL, args, kwargs)
        __call__.__doc__ = doc
        return __call__
    elif name in slicers:                                 # 32/64 bit issue #41
        def method(self, start, stop, *args):
            if stop == maxint:
                stop = None
            return syncreq(self, consts.HANDLE_OLDSLICING, slicers[name], name, start, stop, args)
        method.__name__ = name
        method.__doc__ = doc
        return method
    elif name == "__array__":
        def __array__(self, *args, **kwargs):
            # Note that protocol=-1 will only work between python
            # interpreters of the same version.
            conn = get_live_conn(self)
            if conn is None or not conn._config["allow_pickle"]:
                # Security check that server side allows pickling per #551
                raise ValueError("pickling is disabled")
            array = pickle.loads(syncreq(self, consts.HANDLE_PICKLE, -1))
            return array.__array__(*args, **kwargs)
        __array__.__doc__ = doc
        return __array__
    else:
        def method(_self, *args, **kwargs):
            kwargs = tuple(kwargs.items())
            return syncreq(_self, consts.HANDLE_CALLATTR, name, args, kwargs)
        method.__name__ = name
        method.__doc__ = doc
        return method


class NetrefClass(object):
    """a descriptor of the class being proxied

    Future considerations:
     + there may be a cleaner alternative but lib.compat.with_metaclass prevented using __new__
     + consider using __slot__ for this class
     + revisit the design choice to use properties here
    """

    def __init__(self, class_obj):
        self._class_obj = class_obj

    @property
    def instance(self):
        """accessor to class object for the instance being proxied"""
        return self._class_obj

    @property
    def owner(self):
        """accessor to the class object for the instance owner being proxied"""
        return self._class_obj.__class__

    def __get__(self, netref_instance, netref_owner):
        """the value returned when accessing the netref class is dictated by whether or not an instance is proxied"""
        return self.owner if netref_instance.____id_pack__[2] == 0 else self.instance


def class_factory(id_pack, methods):
    """Creates a netref class proxying the given class

    :param id_pack: the id pack used for proxy communication
    :param methods: a list of ``(method name, docstring)`` tuples, of the methods that the class defines

    :returns: a netref class
    """
    ns = {"__slots__": (), "__class__": None}
    name_pack = id_pack[0]
    class_descriptor = None
    if name_pack is not None:
        # attempt to resolve __class__ using normalized builtins first
        _builtin_class = _normalized_builtin_types.get(name_pack)
        if _builtin_class is not None:
            class_descriptor = NetrefClass(_builtin_class)
        # then by imported modules (this also tries all builtins under "builtins")
        else:
            _module = None
            cursor = len(name_pack)
            while cursor != -1:
                _module = sys.modules.get(name_pack[:cursor])
                if _module is None:
                    cursor = name_pack[:cursor].rfind('.')
                    continue
                _class_name = name_pack[cursor + 1:]
                _class = getattr(_module, _class_name, None)
                if _class is not None and hasattr(_class, '__class__'):
                    class_descriptor = NetrefClass(_class)
                elif _class is None:
                    class_descriptor = NetrefClass(type(_module))
                break
    ns['__class__'] = class_descriptor
    # create methods that must perform a syncreq
    for name, doc in methods:
        name = str(name)  # IronPython issue #10
        # only create methods that won't shadow BaseNetref during merge for mro
        if name not in LOCAL_ATTRS:  # i.e. `name != __class__`
            ns[name] = _make_method(name, doc)
    netref_cls = type(name_pack, (BaseNetref, ), ns)
    return netref_cls


for _builtin in _builtin_types:
    _id_pack = get_id_pack(_builtin)
    _name_pack = _id_pack[0]
    _normalized_builtin_types[_name_pack] = _builtin
    _builtin_methods = get_methods(LOCAL_ATTRS, _builtin)
    # assume all normalized builtins are classes
    builtin_classes_cache[_name_pack] = class_factory(_id_pack, _builtin_methods)
