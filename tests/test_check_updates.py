import requests

from check_updates_and_trigger_run import MARKETPLACE_DISPATCH_URL, notify_marketplace


class FakePost:
    def __init__(self, status_code=204, error=None):
        self.status_code = status_code
        self.error = error
        self.calls = []

    def __call__(self, url, headers=None, json=None, timeout=None):
        self.calls.append((url, headers, json))
        if self.error:
            raise self.error
        return type("Response", (), {"status_code": self.status_code, "text": ""})()


def test_notify_marketplace_sends_benchmarks_updated_dispatch():
    post = FakePost()
    notify_marketplace("secret", post=post)
    [(url, headers, body)] = post.calls
    assert url == MARKETPLACE_DISPATCH_URL
    assert headers["Authorization"] == "Bearer secret"
    assert body == {"event_type": "benchmarks-updated"}


def test_notify_marketplace_does_nothing_without_a_token():
    post = FakePost()
    notify_marketplace(None, post=post)
    assert post.calls == []


def test_notify_marketplace_failure_does_not_raise(caplog):
    notify_marketplace("secret", post=FakePost(status_code=401))
    notify_marketplace("secret", post=FakePost(error=requests.ConnectionError("down")))
    assert caplog.text.count("Could not notify the marketplace") == 2
