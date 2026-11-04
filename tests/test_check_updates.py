from check_updates_and_trigger_run import MARKETPLACE_DISPATCH_URL, notify_marketplace


def test_notify_marketplace_sends_benchmarks_updated_dispatch():
    calls = []

    def post(url, headers=None, json=None, timeout=None):
        calls.append((url, headers, json))
        return type("Response", (), {"status_code": 204, "text": ""})()

    notify_marketplace("secret", post=post)
    assert calls == [
        (
            MARKETPLACE_DISPATCH_URL,
            {"Authorization": "Bearer secret", "Accept": "application/vnd.github+json"},
            {"event_type": "benchmarks-updated"},
        )
    ]
