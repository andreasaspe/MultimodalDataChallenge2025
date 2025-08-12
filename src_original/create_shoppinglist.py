import csv

# Data to be written to the CSV file
data = [
    ["fungi_train000000.jpg", "Habitat"],
    ["fungi_train000002.jpg", "Latitude"],
    ["fungi_train000018.jpg", "Longitude"],
    ["fungi_train000019.jpg", "Substrate"],
    ["fungi_train000018.jpg", "eventDate"],
]

# Specify the output file name
output_file = "shoppinglist.csv"

# Write data to the CSV file
with open(output_file, mode="w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    # Write rows into the file
    writer.writerows(data)

print(f"Data has been successfully written to '{output_file}'.")