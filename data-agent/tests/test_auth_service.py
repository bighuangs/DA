import unittest

from app.application.services.auth_service import create_access_token, decode_access_token, hash_password, verify_password


class AuthServiceTest(unittest.TestCase):
    def test_password_hash_is_not_plaintext_and_verifies(self):
        encoded = hash_password("test-password-123")
        self.assertNotIn("test-password-123", encoded)
        self.assertTrue(verify_password("test-password-123", encoded))
        self.assertFalse(verify_password("wrong-password", encoded))

    def test_access_token_round_trip(self):
        token = create_access_token(user_id="user-1", workspace_id="workspace-1")
        payload = decode_access_token(token)
        self.assertEqual(payload["sub"], "user-1")
        self.assertEqual(payload["workspace_id"], "workspace-1")


if __name__ == "__main__":
    unittest.main()
