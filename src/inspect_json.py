import json

json_path = "/app/data/InstanceBuilding/via_region_data-train.json"

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print("=" * 60)
print("INSTANCEBUILDING VIA JSON INSPECTION")
print("=" * 60)

print("JSON type:", type(data).__name__)

if isinstance(data, dict):

    print("Number of entries:", len(data))

    first_key = next(iter(data))
    first_item = data[first_key]

    print("\nFIRST KEY:")
    print(first_key)

    print("\nFIRST ITEM:")
    print(first_item)

elif isinstance(data, list):

    print("Number of entries:", len(data))

    print("\nFIRST ENTRY:")
    print(data[0])

print("=" * 60)