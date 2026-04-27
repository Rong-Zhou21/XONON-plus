from minerl.herobraine.hero.handlers.translation import TranslationHandler
from minerl.herobraine.hero import spaces

from typing import Dict, Any


class PlainInventoryObservation(TranslationHandler):
    """obs['plain_inventory']"""

    n_slots = 36

    def to_string(self) -> str:
        return "plain_inventory"

    def xml_template(self) -> str:
        return str("""<ObservationFromFullInventory flat="false"/>""")

    def __init__(self):
        shape = (self.n_slots,)
        space = spaces.Dict(
            {
                slot_id: spaces.Dict(
                    {
                        "type": spaces.Text(shape=()),
                        "quantity": spaces.Box(low=0, high=64, shape=()),
                    }
                )
                for slot_id in range(36)
            }
        )
        super().__init__(space=space)

    def from_hero(self, obs_dict: Dict[str, Any]):
        assert "inventory" in obs_dict, "Missing inventory key in malmo json"
        inventory = dict()
        for idx, item in enumerate(obs_dict["inventory"]):
            # Compatibility: MCP-Reborn may emit 'slot_id', 'slot', 'index', or no slot field at all
            if "slot_id" in item:
                slot = item["slot_id"]
            elif "slot" in item:
                slot = item["slot"]
            elif "index" in item:
                slot = item["index"]
            elif "InventorySlot" in item:
                slot = item["InventorySlot"]
            else:
                slot = idx
            try:
                slot = int(slot)
            except (TypeError, ValueError):
                slot = idx
            inventory[slot] = {
                "type": item.get("type", item.get("Type", "air")),
                "quantity": item.get("quantity", item.get("Quantity", item.get("count", 0))),
            }

        return inventory
