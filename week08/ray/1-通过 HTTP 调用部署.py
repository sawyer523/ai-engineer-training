import starlette.requests
import requests
from ray import serve


@serve.deployment
class Counter:
    def __call__(self, request: starlette.requests.Request):
        return request.query_params

serve.run(Counter.bind())

resp = requests.get("http://127.0.0.1:8000?a=b&c=d")
print(resp.json())
assert resp.json() == {"a": "b", "c": "d"}