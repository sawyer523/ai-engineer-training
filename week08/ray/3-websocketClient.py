from websockets.sync.client import connect

with connect("ws://127.0.0.1:8000/") as websocket:
    websocket.send("Eureka!")
    assert websocket.recv() == "Eureka!"

    websocket.send("I've found it!")
    print(websocket.recv())
    assert websocket.recv() == "I've found it!"
    websocket.close()