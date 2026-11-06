import pytest

from classic_precision import natural_t4_x


def test_static_worker_scopes_endpoint_check_and_restores_hooks(monkeypatch):
    route = natural_t4_x
    protocol = {'scope': route.NAME, 'prompt': 'Endpoint prompt'}
    prior_protocol = {'scope': 'T1–T3', 'prompt': 'Original qualification prompt'}
    previous_check = route.static.check_item
    previous_gate = route.static.start_gate
    calls = []

    def original_check(item, candidate):
        calls.append(('original', item, candidate))

    def endpoint_check(item, candidate):
        calls.append(('endpoint', item, candidate))

    def execute(*args, **kwargs):
        assert route.static.start_gate is route.start_gate
        route.static.check_item('natural endpoint pair', dict(protocol))
        route.static.check_item('qualification record', prior_protocol)
        raise RuntimeError('synthetic worker stop')

    with monkeypatch.context() as patch:
        patch.setattr(route, 'read', lambda path: {'modality': 'static', 'static_protocol': 'bound reference'})
        patch.setattr(route, 'verified', lambda ref: protocol if ref == 'bound reference' else None)
        patch.setattr(route, 'check_static', endpoint_check)
        patch.setattr(route.static, 'check_item', original_check)
        patch.setattr(route.static, 'execute_bound_profile', execute)
        with pytest.raises(RuntimeError, match='synthetic worker stop'):
            route.run('unused-profile.json')
        assert route.static.check_item is original_check
        assert route.static.start_gate is previous_gate

    assert route.static.check_item is previous_check
    assert calls == [
        ('endpoint', 'natural endpoint pair', protocol),
        ('original', 'qualification record', prior_protocol),
    ]
