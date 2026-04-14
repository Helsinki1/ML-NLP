

from typing import Any, Dict
from src.envs.tool import Tool


class FindUserIdByNameZip(Tool):
    @staticmethod
    def invoke(data: Dict[str, Any], first_name: str, last_name: str, zip: str) -> str:
        """
        Find a user ID by matching first name, last name, and zip code.

        Args:
            data: The shared database dict containing a "users" key.
            first_name: The customer's first name (case-insensitive match).
            last_name: The customer's last name (case-insensitive match).
            zip: The customer's zip code (exact match).

        Returns:
            The user_id string if a matching user is found, otherwise
            "Error: user not found".
        """

        user_id = None
        ############################################################
        # STUDENT IMPLEMENTATION START
        # Iterate over all users. For each user, check whether their first name,
        # last name (both case-insensitive), and zip code (exact) match the
        # arguments. On the first match, record the user_id and stop.
        ############################################################


        users = data["users"]
        for uid, profile in users.items():
            user_first_name = profile["name"]["first_name"].lower()
            user_last_name = profile["name"]["last_name"].lower()
            first_name = first_name.lower()
            last_name = last_name.lower()
            user_zip = profile["address"]["zip"]
            if user_first_name == first_name and user_last_name == last_name and user_zip == zip:
                user_id = uid


        ############################################################
        # STUDENT IMPLEMENTATION END
        ############################################################

        if user_id is None:
            return "Error: user not found"
        return user_id

    @staticmethod
    def get_info() -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "find_user_id_by_name_zip",
                "description": (
                    "Find user id by first name, last name, and zip code. If the user is not found, the function "
                    "will return an error message. By default, find user id by email, and only call this function "
                    "if the user is not found by email or cannot remember email."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "first_name": {
                            "type": "string",
                            "description": "The first name of the customer, such as 'John'.",
                        },
                        "last_name": {
                            "type": "string",
                            "description": "The last name of the customer, such as 'Doe'.",
                        },
                        "zip": {
                            "type": "string",
                            "description": "The zip code of the customer, such as '12345'.",
                        },
                    },
                    "required": ["first_name", "last_name", "zip"],
                },
            },
        }
