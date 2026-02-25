# Extract McCullough Customers by City (Red Rock & Georgetown)

## Quick Start

```bash
python scripts/extract_customers_by_city.py
```

Output: `data/McCullough Heating and Air/red_rock_georgetown_customers.csv`

## Custom Options

```bash
# Custom cities
python scripts/extract_customers_by_city.py --cities "Red Rock" "Georgetown" "Round Rock"

# Custom output file
python scripts/extract_customers_by_city.py -o output/my_customers.csv

# Custom parquet directory
python scripts/extract_customers_by_city.py --parquet-dir path/to/parquet
```

## How It Works

1. **Customers.parquet** – Filters rows where:
   - `City` (or similar) column equals Red Rock or Georgetown
   - `Full Address` / `Customer Address` contains "Red Rock" or "Georgetown"

2. **Locations.parquet** – If present, also includes customers whose *locations* are in these cities (in case the customer record doesn’t have city but a location does).

## Required Data

- `companies/McCullough Heating and Air/parquet/Customers.parquet`
- `companies/McCullough Heating and Air/parquet/Locations.parquet` (optional, adds location-based matches)

Ensure the Data Pipeline has been run so these parquet files exist (from the Gmail/ServiceTitan export).

## Alternative: Use the App

To filter in the Streamlit app, you’d need to add a city filter to the Data Pipeline or Events page. The script is a standalone alternative for one-off exports.
