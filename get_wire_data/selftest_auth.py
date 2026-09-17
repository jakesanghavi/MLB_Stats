"""Offline checks for mlb_auth + a live login probe.

The offline checks (PKCE S256, JWT exp parsing, token caching) need no network.
The live probe actually attempts the login; from a blocked network it reports
the HTTP 451 rather than failing the whole script.
"""
import base64
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mlb_auth import MlbAuth, make_pkce, jwt_exp, AuthBlockedError, AuthError


def _b64url(raw):
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def test_pkce_known_vector():
    # RFC 7636 Appendix B
    verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"
    expected = "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
    got = _b64url(hashlib.sha256(verifier.encode()).digest())
    assert got == expected, got
    # our generator returns a valid pair with matching S256 relationship
    v, c = make_pkce()
    assert 43 <= len(v) <= 128
    assert c == _b64url(hashlib.sha256(v.encode()).digest())
    print("PKCE S256: OK")


def test_jwt_exp():
    payload = _b64url(json.dumps({"exp": 1775582700, "aud": "api://mlb_default"}).encode())
    token = "hdr." + payload + ".sig"
    assert jwt_exp(token) == 1775582700
    assert jwt_exp("garbage") == 0
    print("jwt_exp parsing: OK")


def test_caching():
    a = MlbAuth(username="u", password="p")
    calls = {"n": 0}

    def fake_login():
        calls["n"] += 1
        a._token = "tok-%d" % calls["n"]
        a._exp = int(time.time()) + 3600
        return a._token

    a.login = fake_login
    assert a.token() == "tok-1"
    assert a.token() == "tok-1"          # cached, no second login
    a._exp = int(time.time()) + 10       # force near-expiry
    assert a.token() == "tok-2"          # re-login
    assert calls["n"] == 2
    print("token caching/refresh: OK")


def live_login_probe():
    print("\n-- live login probe --")
    try:
        tok = MlbAuth().login()
        claims_b = tok.split(".")[1]; claims_b += "=" * (-len(claims_b) % 4)
        claims = json.loads(base64.urlsafe_b64decode(claims_b))
        print("LOGIN OK: token len=%d aud=%s exp_in=%ds entitlement=%s" % (
            len(tok), claims.get("aud"),
            claims.get("exp", 0) - int(time.time()), claims.get("mlb_entitlement")))
        return tok
    except AuthBlockedError as e:
        print("BLOCKED (expected from datacenter IP):", e)
    except AuthError as e:
        print("AUTH ERROR:", e)
    return None


if __name__ == "__main__":
    test_pkce_known_vector()
    test_jwt_exp()
    test_caching()
    print("offline auth checks passed")
    live_login_probe()
