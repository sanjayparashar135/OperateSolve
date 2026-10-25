"""Conversation CRUD."""


def test_create_list_rename_and_delete(client):
    created = client.post("/api/chats", json={})
    assert created.status_code == 201
    chat = created.json()
    assert chat["title"] == "New conversation"
    assert chat["title_customized"] is False

    listed = client.get("/api/chats")
    assert listed.status_code == 200
    assert listed.json()[0]["id"] == chat["id"]

    renamed = client.patch(f"/api/chats/{chat['id']}", json={"title": "  Checkout incident  "})
    assert renamed.status_code == 200
    assert renamed.json()["title"] == "Checkout incident"
    assert renamed.json()["title_customized"] is True

    missing = client.get("/api/chats/missing-conversation")
    assert missing.status_code == 404

    deleted = client.delete(f"/api/chats/{chat['id']}")
    assert deleted.status_code == 204
    assert client.get(f"/api/chats/{chat['id']}").status_code == 404


def test_blank_title_is_rejected(client):
    created = client.post("/api/chats", json={})
    chat_id = created.json()["id"]
    response = client.patch(f"/api/chats/{chat_id}", json={"title": "   "})
    assert response.status_code == 422
