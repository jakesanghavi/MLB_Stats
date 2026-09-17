"""
Basic programmatic MLB login to mint the bearer token used by the Gameday 3D
(mannequin/FieldVision) tracking API.

Flow (Okta Authorization Code + PKCE), reconstructed from the public
``mlb-okta`` bundle:

  1. POST {okta}/api/v1/authn            {username, password}         -> sessionToken
  2. GET  {issuer}/v1/authorize          (PKCE, sessionToken)         -> 302 ?code=...
  3. POST {issuer}/v1/token              (code + code_verifier)       -> access_token

Config (from the bundle, production):
  clientId  = 0oap7wa857jcvPlZ5355
  issuer    = https://ids.mlb.com/oauth2/aus1m088yK07noBfh356
  redirect  = https://www.mlb.com/login
  scopes    = openid email          (no offline_access -> no refresh token,
                                      so we simply re-login when the token nears exp)

Credentials come from MLB_USERNAME / MLB_PASSWORD (env / secrets); never hard-code them.

NOTE: the login host ``ids.mlb.com`` geo/IP-blocks many datacenter/cloud IPs
(HTTP 451). If ``login()`` raises AuthBlockedError, run from an allowed network
(e.g. a US residential IP) or supply a token minted elsewhere via
MLB_BEARER_TOKEN.
"""

import base64
import hashlib
import json
import os
import secrets
import time
from urllib.parse import urlparse, parse_qs

import requests

OKTA_HOST = "https://ids.mlb.com"
AUTH_SERVER_ID = "aus1m088yK07noBfh356"
ISSUER = f"{OKTA_HOST}/oauth2/{AUTH_SERVER_ID}"
CLIENT_ID = "0oap7wa857jcvPlZ5355"
REDIRECT_URI = "https://www.mlb.com/login"
SCOPES = "openid email"
_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36")


class AuthError(RuntimeError):
    pass


class AuthBlockedError(AuthError):
    """Raised when MLB returns HTTP 451 (source IP not permitted to log in)."""


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def make_pkce():
    """Return (code_verifier, code_challenge) for PKCE S256."""
    verifier = _b64url(secrets.token_bytes(32))          # 43 chars, RFC 7636 compliant
    challenge = _b64url(hashlib.sha256(verifier.encode("ascii")).digest())
    return verifier, challenge


def jwt_exp(token: str) -> int:
    """Return the ``exp`` (unix seconds) from a JWT without verifying it."""
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        return int(json.loads(base64.urlsafe_b64decode(payload)).get("exp", 0))
    except Exception:
        return 0


class MlbAuth:
    """Caches an access token and re-logs-in when it is missing or near expiry."""

    def __init__(self, username=None, password=None, *, client_id=CLIENT_ID,
                 issuer=ISSUER, redirect_uri=REDIRECT_URI, scopes=SCOPES,
                 early_seconds=60, session=None):
        self.username = username or os.environ.get("MLB_USERNAME")
        self.password = password or os.environ.get("MLB_PASSWORD")
        self.client_id = client_id
        self.issuer = issuer
        self.redirect_uri = redirect_uri
        self.scopes = scopes
        self.early_seconds = early_seconds
        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", _UA)
        self._token = None
        self._exp = 0

    # -- public API -------------------------------------------------------
    def token(self):
        """Return a valid access token, logging in / refreshing as needed."""
        if self._token and time.time() < self._exp - self.early_seconds:
            return self._token
        return self.login()

    def login(self):
        if not self.username or not self.password:
            raise AuthError("MLB_USERNAME / MLB_PASSWORD not set")
        session_token = self._authn()
        code, verifier = self._authorize(session_token)
        access_token = self._exchange(code, verifier)
        self._token = access_token
        self._exp = jwt_exp(access_token) or (int(time.time()) + 3600)
        return access_token

    # -- steps ------------------------------------------------------------
    def _authn(self):
        r = self.session.post(
            f"{OKTA_HOST}/api/v1/authn",
            json={"username": self.username, "password": self.password,
                  "options": {"multiOptionalFactorEnroll": False,
                              "warnBeforePasswordExpired": False}},
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            timeout=30, allow_redirects=False,
        )
        self._raise_if_blocked(r, "authn")
        try:
            j = r.json()
        except ValueError:
            raise AuthError(f"authn returned non-JSON (status {r.status_code})")
        status = j.get("status")
        if status != "SUCCESS":
            if status == "MFA_REQUIRED":
                raise AuthError("account has MFA enabled; unattended login unsupported")
            raise AuthError(f"authn status={status}")
        return j["sessionToken"]

    def _authorize(self, session_token):
        verifier, challenge = make_pkce()
        params = {
            "client_id": self.client_id, "response_type": "code", "scope": self.scopes,
            "redirect_uri": self.redirect_uri, "state": secrets.token_hex(8),
            "nonce": secrets.token_hex(8), "code_challenge": challenge,
            "code_challenge_method": "S256", "sessionToken": session_token,
        }
        r = self.session.get(f"{self.issuer}/v1/authorize", params=params,
                             allow_redirects=False, timeout=30)
        self._raise_if_blocked(r, "authorize")
        loc = r.headers.get("Location", "")
        q = parse_qs(urlparse(loc).query)
        if "code" not in q:
            raise AuthError(f"authorize did not return a code (status {r.status_code}, "
                            f"error={q.get('error')})")
        return q["code"][0], verifier

    def _exchange(self, code, verifier):
        r = self.session.post(
            f"{self.issuer}/v1/token",
            data={"grant_type": "authorization_code", "code": code,
                  "code_verifier": verifier, "client_id": self.client_id,
                  "redirect_uri": self.redirect_uri},
            headers={"Accept": "application/json",
                     "Content-Type": "application/x-www-form-urlencoded"},
            timeout=30, allow_redirects=False,
        )
        self._raise_if_blocked(r, "token")
        try:
            j = r.json()
        except ValueError:
            raise AuthError(f"token endpoint returned non-JSON (status {r.status_code})")
        if "access_token" not in j:
            raise AuthError(f"token exchange failed: {j.get('error')} {j.get('error_description')}")
        return j["access_token"]

    @staticmethod
    def _raise_if_blocked(resp, step):
        if resp.status_code == 451:
            raise AuthBlockedError(
                f"{step}: HTTP 451 from ids.mlb.com (this IP is not permitted to log in). "
                "Run from an allowed network or supply MLB_BEARER_TOKEN.")


def bearer_token():
    """Convenience: mint a token from MLB_USERNAME/MLB_PASSWORD env vars."""
    return MlbAuth().token()
