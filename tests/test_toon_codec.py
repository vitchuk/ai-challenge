"""Тесты TOON-кодека (обёртки над toon_format)."""

from server import toon_codec


def test_roundtrip_object():
    data = {"name": "Alice", "age": 30, "active": True}
    assert toon_codec.decode(toon_codec.encode(data)) == data


def test_roundtrip_list_of_dicts():
    data = [{"id": 1, "name": "A"}, {"id": 2, "name": "B"}]
    assert toon_codec.decode(toon_codec.encode(data)) == data


def test_encode_is_compact_text():
    encoded = toon_codec.encode({"users": [{"id": 1, "name": "Ada"}]})
    assert isinstance(encoded, str)
    assert "users" in encoded
    # TOON не использует JSON-объектные скобки для хранения (табличный вид);
    # проверяем лишь, что это компактный текст без кавычек/скобок массива JSON
    assert "[" not in encoded.replace("users[1]", "")
    assert '"' not in encoded
