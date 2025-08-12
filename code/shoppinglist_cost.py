import csv

# Prices
metadata_prices = {
    "eventDate": 2,
    "Latitude": 1,
    "Longitude": 1,
    "Habitat": 2,
    "Substrate": 2
}

# Parameters
num_records = 1000  # how many images
all_metadata = list(metadata_prices.keys())

output_file = "shoppinglist.csv"
total_cost = 0
data = []

for i in range(num_records):
    image_name = f"fungi_train{i:06d}.jpg"
    for meta in all_metadata:
        data.append([image_name, meta])
        total_cost += metadata_prices[meta]

# Write CSV
with open(output_file, mode="w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerows(data)

print(f"Shopping list saved to '{output_file}'")
print(f"Total entries: {len(data)}")
print(f"Total cost: {total_cost} credits")
