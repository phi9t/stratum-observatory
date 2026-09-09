import base64,copy,hashlib,json,subprocess,tempfile,unittest
from unittest import mock
from datetime import datetime,timezone,timedelta
from pathlib import Path
from verification.evidence import canonical,check_review,ed25519_openssl
class SignedReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        openssl=ed25519_openssl()
        if not openssl:self.skipTest('No OpenSSL with Ed25519; install OpenSSL 3.x or set STRATUM_OPENSSL.')
        self.openssl=openssl
        subprocess.run([openssl,'genpkey','-algorithm','Ed25519','-out',str(self.root/'private.pem')],check=True,capture_output=True)
        subprocess.run([openssl,'pkey','-in',str(self.root/'private.pem'),'-pubout','-out',str(self.root/'reviewer.pem')],check=True,capture_output=True)
        self.trust={'reviewers':{'independent':{'public_key':'reviewer.pem'}}}
        self.body={'author':'implementer','reviewer':'independent','subject_sha256':'a'*64,'verdict':'accept','expires_at':(datetime.now(timezone.utc)+timedelta(days=7)).isoformat()}
    def sign(self,body):
        (self.root/'body').write_bytes(canonical(body))
        subprocess.run([self.openssl,'pkeyutl','-sign','-inkey',str(self.root/'private.pem'),'-rawin','-in',str(self.root/'body'),'-out',str(self.root/'signature')],check=True,capture_output=True)
        return {'c':{'body':body,'signature':base64.b64encode((self.root/'signature').read_bytes()).decode()}}
    def check(self,reviews,trust=None):return check_review({'id':'c'},'a'*64,reviews,self.root,self.trust if trust is None else trust)
    def test_real_ed25519_acceptance(self):self.assertEqual(self.check(self.sign(self.body)).status,'PASS')
    def test_tampered_body_rejected(self):
        r=self.sign(copy.deepcopy(self.body));r['c']['body']['author']='changed'
        self.assertEqual(self.check(r).status,'FAIL')
    def test_changed_evidence_digest_rejected(self):
        body={**self.body,'subject_sha256':'b'*64}
        self.assertEqual(self.check(self.sign(body)).status,'FAIL')
    def test_self_review_rejected(self):
        self.assertEqual(self.check(self.sign({**self.body,'author':'independent'})).status,'FAIL')
    def test_expired_review_rejected(self):
        self.assertEqual(self.check(self.sign({**self.body,'expires_at':'2020-01-01T00:00:00Z'})).status,'FAIL')
    def test_unknown_reviewer_blocked(self):self.assertEqual(self.check(self.sign(self.body),{}).status,'BLOCKED')
    def test_revoked_key_blocked(self):
        t=copy.deepcopy(self.trust);t['reviewers']['independent']['revoked']=True
        self.assertNotEqual(self.check(self.sign(self.body),t).status,'PASS')


class ToolchainCapabilityTests(unittest.TestCase):
    """A verifier host without Ed25519 must block, never contradict a real reviewer."""
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        (self.root/'reviewer.pem').write_text('unused: the signature is never reached without a capable toolchain\n')
        self.trust={'reviewers':{'independent':{'public_key':'reviewer.pem'}}}
        self.review={'c':{'body':{'author':'implementer','reviewer':'independent','subject_sha256':'a'*64,'verdict':'accept',
                                 'expires_at':(datetime.now(timezone.utc)+timedelta(days=7)).isoformat()},
                           'signature':base64.b64encode(b'\0'*64).decode()}}
    def test_missing_ed25519_toolchain_blocks_and_does_not_fail(self):
        with mock.patch('verification.evidence.ed25519_openssl',return_value=None):
            finding=check_review({'id':'c'},'a'*64,self.review,self.root,self.trust)
        self.assertEqual(finding.status,'BLOCKED')
        self.assertEqual(finding.code,'REVIEW_TOOLCHAIN')
    def test_capable_toolchain_is_detected_on_this_host(self):
        from verification.evidence import ed25519_openssl
        self.assertIsNotNone(ed25519_openssl(),'No OpenSSL with Ed25519 found; install OpenSSL 3.x (LibreSSL cannot verify these reviews).')
