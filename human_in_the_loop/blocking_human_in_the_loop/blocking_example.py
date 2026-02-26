from google.adk import Agent
from google.adk.tools import FunctionTool


def calculate_total(order_items: dict[str, int]) -> int:
  PRICES = {'burger': 10, 'fry': 5, 'soda': 3}
  total = 0
  for item, quantity in order_items.items():
    total += quantity * PRICES[item]
  return total


def place_order(order_items: dict[str, int]):
  """Places an order for the given items.

  Args:
    order_items: The items to be ordered. A dict of item name to quantity. The
      item name can be 'burger', 'fry', or 'soda'.
  """
  total = calculate_total(order_items)
  return {'status': 'success', 'order_items': order_items, 'total': total}

def confirmation_criteria(order_items: dict[str, int]) -> bool:
  return calculate_total(order_items) > 100

root_agent = Agent(
    model='gemini-2.5-flash',
    name='fast_food_agent',
    instruction="""You help customers place orders at a restaurant.""",
    tools=[
        FunctionTool(place_order, require_confirmation=confirmation_criteria),
    ],
)