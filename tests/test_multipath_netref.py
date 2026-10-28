"""Tests for multi-path netrefs over independent RPyC relay connections.

Scenario (all services run in this process on dedicated threads)::

    client process
       |-- c1 --> relay R1 --u1--+
       |                         |--> source server S: one Resource object
       |-- c2 --> relay R2 --u2--+

Requirements covered:

* the same source object fetched over the two independent relay chains is
  represented by the *same* proxy object in the client process;
* calls and bidirectional callbacks may travel over any live path;
* shutting down one relay (path) does not affect the other path;
* the source object is only reclaimed once *all* paths have been released.
"""
import gc
import time
import weakref
import unittest

import rpyc
from rpyc.core import netref
from rpyc.utils.server import ThreadedServer


SOURCE_PORT = 18911
RELAY1_PORT = 18912
RELAY2_PORT = 18913

# Logical RPyC nodes. In a real deployment each node is a separate process and
# needs no explicit scope; scopes are supplied here because every node is
# emulated inside this one test process.
CLIENT_SCOPE = netref.NetrefScope()
SOURCE_SCOPE = netref.NetrefScope()
RELAY1_SCOPE = netref.NetrefScope()
RELAY2_SCOPE = netref.NetrefScope()


# populated/reset by the tests; the servers below run in-process, so the
# resource lifecycle is directly observable
_resource_ref = None
_resource_deleted = []


class Resource(object):
    """The single source-side object reached through both relay chains."""

    def __init__(self):
        self.value = 0
        self.callbacks = []

    def exposed_add(self, x):
        self.value += x
        return self.value

    def exposed_echo(self, x):
        return x

    def exposed_register_callback(self, cb):
        self.callbacks.append(cb)

    def exposed_fire_callbacks(self, x):
        # tuple is passed by value (LABEL_TUPLE), a list would come back as
        # another netref
        return tuple(cb(x) for cb in self.callbacks)

    def exposed_callback_count(self):
        return len(self.callbacks)

    def exposed_callbacks_share_identity(self):
        return len(self.callbacks) >= 2 and self.callbacks[0] is self.callbacks[1]

    def __del__(self):
        _resource_deleted.append(id(self))


class SourceService(rpyc.Service):
    """Serves the one-and-only Resource, held only by a weak reference here.

    Strong references while paths are alive live exclusively in the
    connection-local ``_local_objects`` collections, so observing the
    weakref reports exactly when the source object may be reclaimed.
    """

    PORT = SOURCE_PORT

    def exposed_get_resource(self):
        global _resource_ref
        resource = _resource_ref() if _resource_ref is not None else None
        if resource is None:
            resource = Resource()
            _resource_ref = weakref.ref(resource)
        return resource


class _RelayService(rpyc.Service):
    """Opens a dedicated upstream connection per inbound connection and
    closes it on disconnect, making the two relay chains independent."""

    def on_connect(self, conn):
        # the upstream link belongs to the same relay node as the inbound link
        self._upstream = rpyc.connect("localhost", port=SOURCE_PORT,
                                      config={"netref_scope": conn._netref_scope})

    def on_disconnect(self, conn):
        try:
            self._upstream.close()
        except Exception:
            pass

    def exposed_get_resource(self):
        # beyond this hop the resource is a netref and every attribute access
        # or call is forwarded through the upstream connection transparently
        return self._upstream.root.get_resource()


class RelayService1(_RelayService):
    PORT = RELAY1_PORT


class RelayService2(_RelayService):
    PORT = RELAY2_PORT


def _wait_until(condition, timeout=5.0, interval=0.02):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if condition():
            return
        time.sleep(interval)
    raise AssertionError("condition not met within timeout")


class TestMultipathRelay(unittest.TestCase):

    def setUp(self):
        global _resource_ref, _resource_deleted
        _resource_ref = None
        _resource_deleted = []

        self.source = ThreadedServer(SourceService, port=SOURCE_PORT, auto_register=False,
                                     protocol_config={"netref_scope": SOURCE_SCOPE})
        self.relay1 = ThreadedServer(RelayService1, port=RELAY1_PORT, auto_register=False,
                                     protocol_config={"netref_scope": RELAY1_SCOPE})
        self.relay2 = ThreadedServer(RelayService2, port=RELAY2_PORT, auto_register=False,
                                     protocol_config={"netref_scope": RELAY2_SCOPE})
        for server in (self.source, self.relay1, self.relay2):
            server._start_in_thread()

        self.c1 = rpyc.connect("localhost", RELAY1_PORT,
                               config={"netref_scope": CLIENT_SCOPE})
        self.c2 = rpyc.connect("localhost", RELAY2_PORT,
                               config={"netref_scope": CLIENT_SCOPE})
        # background serving threads let reverse callbacks arrive on either
        # connection while the main thread waits on a synchronous call
        self.bg1 = rpyc.BgServingThread(self.c1)
        self.bg2 = rpyc.BgServingThread(self.c2)

    def tearDown(self):
        for bg in (self.bg1, self.bg2):
            if bg._active:
                bg.stop()
        for conn in (self.c1, self.c2):
            if not conn.closed:
                conn.close()
        for server in (self.source, self.relay1, self.relay2):
            _wait_until(lambda s=server: not s.clients, timeout=5)
            server.close()

    @staticmethod
    def _close_path(conn, bg):
        # stop background serving before the stream is torn down, otherwise
        # the serving thread observes the closed stream and re-raises
        if bg._active:
            bg.stop()
        conn.close()

    def test_same_proxy_over_two_relay_chains(self):
        r1 = self.c1.root.get_resource()
        r2 = self.c2.root.get_resource()
        self.assertIs(r1, r2, "both relay paths must yield the same proxy")
        self.assertTrue(netref.has_route(r1, self.c1))
        self.assertTrue(netref.has_route(r1, self.c2))

    def test_calls_can_take_either_path(self):
        r1 = self.c1.root.get_resource()
        r2 = self.c2.root.get_resource()
        self.assertIs(r1, r2)

        self.assertEqual(r1.add(5), 5)
        self.assertEqual(r2.add(7), 12)  # cumulative state: same source object

        # explicitly pin the preferred route to the second connection
        object.__setattr__(r1, "____conn__", self.c2)
        self.assertEqual(r1.add(30), 42)
        self.assertIs(netref.get_live_conn(r1), self.c2)

    def test_closing_one_relay_keeps_proxy_alive(self):
        r1 = self.c1.root.get_resource()
        r2 = self.c2.root.get_resource()
        self.assertIs(r1, r2)
        self.assertEqual(r1.add(1), 1)
        self.assertIsNotNone(_resource_ref())

        self._close_path(self.c1, self.bg1)
        # path 1 (client->R1->S) is gone, including the source-side socket
        _wait_until(lambda: not netref.has_route(r1, self.c1))
        _wait_until(lambda: len(self.source.clients) == 1)

        # the shared proxy transparently keeps serving over path 2
        self.assertIs(netref.get_live_conn(r1), self.c2)
        self.assertEqual(r1.add(41), 42)
        self.assertEqual(r2.echo("ok"), "ok")
        # source object is still pinned by the surviving path
        self.assertIsNotNone(_resource_ref())
        self.assertEqual(_resource_deleted, [])

    def test_bidirectional_callbacks_over_any_path(self):
        r1 = self.c1.root.get_resource()
        r2 = self.c2.root.get_resource()
        self.assertIs(r1, r2)

        received = []

        def cb(x):
            received.append(x)
            return x + 1

        # deliver one registration through each relay chain (the shared proxy
        # uses one preferred connection, so pin it explicitly)
        object.__setattr__(r1, "____conn__", self.c1)
        r1.register_callback(cb)
        object.__setattr__(r1, "____conn__", self.c2)
        r2.register_callback(cb)
        object.__setattr__(r1, "____conn__", self.c1)
        _wait_until(lambda: r1.callback_count() == 2)
        # the source merged both deliveries into one shared callback proxy:
        # same end-to-end id_pack, same source-side scope
        self.assertTrue(r1.callbacks_share_identity())

        # both paths alive: the callback proxy may be invoked back over either
        # route (its preferred route here is path 1)
        before = len(received)
        self.assertEqual(r2.fire_callbacks(10), (11, 11))
        self.assertEqual(received[before:], [10, 10])

        # tear down path 1, including its source-side upstream connection
        self._close_path(self.c1, self.bg1)
        _wait_until(lambda: len(self.source.clients) == 1)
        _wait_until(lambda: not netref.has_route(r1, self.c1))

        # every callback invocation now reaches the client through path 2
        self.assertIs(netref.get_live_conn(r1), self.c2)
        before = len(received)
        self.assertEqual(r2.fire_callbacks(20), (21, 21))
        self.assertEqual(received[before:], [20, 20])

    def test_source_released_only_after_all_paths_closed(self):
        # closing one path while the proxy is alive keeps the source object
        r1 = self.c1.root.get_resource()
        r2 = self.c2.root.get_resource()
        self.assertIs(r1, r2)
        self.assertIsNotNone(_resource_ref())

        self._close_path(self.c1, self.bg1)
        _wait_until(lambda: len(self.source.clients) == 1)
        gc.collect()
        self.assertIsNotNone(_resource_ref(), "one path still holds the source object")

        self._close_path(self.c2, self.bg2)
        _wait_until(lambda: len(self.source.clients) == 0)
        _wait_until(lambda: _resource_ref() is None)
        self.assertEqual(len(_resource_deleted), 1)
        del r1, r2

    def test_source_released_on_proxy_dealloc_while_paths_alive(self):
        # releasing the shared proxy sends HANDLE_DEL over every route;
        # the source object becomes collectable without any relay closing
        r1 = self.c1.root.get_resource()
        r2 = self.c2.root.get_resource()
        self.assertIs(r1, r2)
        self.assertIsNotNone(_resource_ref())

        del r1, r2
        gc.collect()
        _wait_until(lambda: _resource_ref() is None)
        self.assertEqual(len(_resource_deleted), 1)


class TestMultipathDirect(unittest.TestCase):
    """Identity guarantee without relays: two connections to the same server
    also share a proxy for the same source object."""

    def setUp(self):
        global _resource_ref, _resource_deleted
        _resource_ref = None
        _resource_deleted = []
        self.server = ThreadedServer(SourceService, port=SOURCE_PORT, auto_register=False,
                                     protocol_config={"netref_scope": SOURCE_SCOPE})
        self.server._start_in_thread()
        self.c1 = rpyc.connect("localhost", SOURCE_PORT,
                               config={"netref_scope": CLIENT_SCOPE})
        self.c2 = rpyc.connect("localhost", SOURCE_PORT,
                               config={"netref_scope": CLIENT_SCOPE})

    def tearDown(self):
        for conn in (self.c1, self.c2):
            if not conn.closed:
                conn.close()
        _wait_until(lambda: not self.server.clients, timeout=5)
        self.server.close()

    def test_shared_proxy_between_two_connections(self):
        r1 = self.c1.root.get_resource()
        r2 = self.c2.root.get_resource()
        self.assertIs(r1, r2)
        self.assertTrue(netref.has_route(r1, self.c1))
        self.assertTrue(netref.has_route(r2, self.c2))
        self.assertEqual(r1.add(2), 2)
        self.assertEqual(r2.add(40), 42)


class TestDefaultScopePrivate(unittest.TestCase):
    """Without an explicit scope every connection keeps its own proxy, the
    historical per-connection behavior."""

    def setUp(self):
        global _resource_ref, _resource_deleted
        _resource_ref = None
        _resource_deleted = []
        self.server = ThreadedServer(SourceService, port=SOURCE_PORT, auto_register=False)
        self.server._start_in_thread()
        self.c1 = rpyc.connect("localhost", SOURCE_PORT)
        self.c2 = rpyc.connect("localhost", SOURCE_PORT)

    def tearDown(self):
        for conn in (self.c1, self.c2):
            if not conn.closed:
                conn.close()
        _wait_until(lambda: not self.server.clients, timeout=5)
        self.server.close()

    def test_proxies_are_not_shared_by_default(self):
        r1 = self.c1.root.get_resource()
        r2 = self.c2.root.get_resource()
        self.assertIsNot(r1, r2)
        # each proxy is routed over its own connection only
        self.assertTrue(netref.has_route(r1, self.c1))
        self.assertFalse(netref.has_route(r1, self.c2))
        self.assertTrue(netref.has_route(r2, self.c2))
        self.assertFalse(netref.has_route(r2, self.c1))


if __name__ == "__main__":
    unittest.main()
