

import json
from typing import Any, Dict
from src.envs.tool import Tool
from src.envs.retail.tools.types import PaymentRecord, TransactionType


class CancelPendingOrder(Tool):
    @staticmethod
    def invoke(data: Dict[str, Any], order_id: str, reason: str) -> str:
        """
        Cancel a pending order and process refunds.

        Args:
            data: The shared database dict with "orders" and "users" keys.
            order_id: The order ID to cancel (e.g. "#W0000000").
            reason: The cancellation reason; must be one of
                    "no longer needed" or "ordered by mistake".

        Returns:
            json.dumps(order) with the updated order on success, or an error
            string on failure.
        """

        # Check order exists
        orders = data["orders"]
        if order_id not in orders:
            return "Error: order not found"
        
        # Check order status
        order = orders[order_id]
        if order["status"] != "pending":
            return "Error: non-pending order cannot be cancelled"

        # Validate reason
        if reason not in ["no longer needed", "ordered by mistake"]:
            return "Error: invalid reason"

        refunds = []
        ############################################################
        # STUDENT IMPLEMENTATION START
        # Build a list of refund records using PaymentRecord.
        # For every entry in order["payment_history"], create:
        #       PaymentRecord(
        #           transaction_type=TransactionType.REFUND,
        #           amount=<same amount as the original payment>,
        #           payment_method_id=<same payment_method_id>,
        #       )
        #    If payment_method_id contains "gift_card", immediately add
        #    the refund amount to the gift card balance in
        #    data["users"][order["user_id"]]["payment_methods"][payment_method_id]["balance"]
        #    and round to 2 decimal places.
        ############################################################


        payment_record = PaymentRecord(
            transaction_type=TransactionType.REFUND,
            amount=order["payment_history"][0]["amount"],
            payment_method_id=order["payment_history"][0]["payment_method_id"],
        )

        refunds.append(payment_record)

        if "gift_card" in payment_record.payment_method_id:
            data["users"][order["user_id"]]["payment_methods"][payment_record.payment_method_id]["balance"] += payment_record.amount
            data["users"][order["user_id"]]["payment_methods"][payment_record.payment_method_id]["balance"] = round(data["users"][order["user_id"]]["payment_methods"][payment_record.payment_method_id]["balance"], 2)


        ############################################################
        # STUDENT IMPLEMENTATION END
        ############################################################

        # update order status
        order["status"] = "cancelled"
        order["cancel_reason"] = reason
        order["payment_history"].extend(r.to_dict() for r in refunds)
        
        return json.dumps(order)

    @staticmethod
    def get_info() -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "cancel_pending_order",
                "description": (
                    "Cancel a pending order. If the order is already processed or delivered, "
                    "it cannot be cancelled. The agent needs to explain the cancellation detail "
                    "and ask for explicit user confirmation (yes/no) to proceed. If the user confirms, "
                    "the order status will be changed to 'cancelled' and the payment will be refunded. "
                    "The refund will be added to the user's gift card balance immediately if the payment "
                    "was made using a gift card, otherwise the refund would take 5-7 business days to process. "
                    "The function returns the order details after the cancellation."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {
                            "type": "string",
                            "description": "The order id, such as '#W0000000'. Be careful there is a '#' symbol at the beginning of the order id.",
                        },
                        "reason": {
                            "type": "string",
                            "enum": ["no longer needed", "ordered by mistake"],
                            "description": "The reason for cancellation, which should be either 'no longer needed' or 'ordered by mistake'.",
                        },
                    },
                    "required": ["order_id", "reason"],
                },
            },
        }
