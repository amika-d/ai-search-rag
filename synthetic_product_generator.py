import dspy 
import os
from pathlib import Path
import json

api_base = "http://localhost:11434"
apikey = os.getenv('DEEPSEEK_API_KEY')

model_type = input("Enter local or remote model type (local/remote): ").strip().lower()
if model_type == 'local':
    lm = dspy.LM('ollama/mistral', api_base=api_base)
elif model_type == 'remote':
    lm = dspy.LM('deepseek/deepseek-chat', api_key=apikey)
    
dspy.configure(lm=lm)

class BulkProductGenerator(dspy.Signature):
    """Generate multiple skincare products in a single call"""
    count: int = dspy.InputField(desc="Number of products to generate")
    instruction: str = dspy.InputField(desc="Instructions for product generation")
    products_json: str = dspy.OutputField(desc="JSON array of all products")

generate_products = dspy.ChainOfThought(BulkProductGenerator)

# Product generation instructions
instruction = """Generate 50 unique skincare products as a JSON array. Each product should follow this structure:
{
  "id": "prod_1" (increment for each product),
  "collection_id": "collection-name" (vary: e.g., "hydration", "anti-aging", "acne-care", "brightening"),
  "name": "Product Name" (creative and unique),
  "description": "Detailed description (2-3 sentences)",
  "price": float (between 15.00 and 250.00),
  "cost_price": float (calculate based on profit margin),
  "profit_margin": float (vary between 30.00 and 75.00),
  "inventory": int (between 20 and 500),
  "rating": float (between 3.5 and 5.0),
  "ingredients": [list of 3-6 key ingredients],
  "concerns_addressed": [list of 2-4 skin concerns like "Dryness", "Aging", "Acne", "Hyperpigmentation"],
  "texture": "texture type (Cream, Gel, Serum, Oil, Lotion, Foam, Balm, Mist)",
  "image": "/products/product-slug.jpg"
}

IMPORTANT RULES:
1. Vary profit margins realistically:
   - High-end products (>$100): 55-75% margin
   - Mid-range ($40-100): 45-60% margin
   - Budget (<$40): 30-50% margin

2. Use realistic skincare ingredients: Hyaluronic Acid, Niacinamide, Retinol, Vitamin C, Ceramides, Peptides, AHA/BHA, Squalane, etc.

3. Create diverse product types: cleansers, toners, serums, moisturizers, masks, sunscreens, eye creams, treatments

4. Make each product COMPLETELY UNIQUE with different names, descriptions, and ingredient combinations

5. Return ONLY valid JSON array starting with [ and ending with ], no markdown formatting or explanations"""

print("Generating 50 products in a single API call...\n")

try:
    result = generate_products(count=50, instruction=instruction)
    json_str = result.products_json
    
    # Clean up the response
    if "```json" in json_str:
        json_str = json_str.split("```json")[1].split("```")[0].strip()
    elif "```" in json_str:
        json_str = json_str.split("```")[1].split("```")[0].strip()
    
    # Remove any leading/trailing whitespace
    json_str = json_str.strip()
    
    # Parse JSON array
    products = json.loads(json_str)
    
    # Validate it's a list
    if not isinstance(products, list):
        raise ValueError("Expected a JSON array of products")
    
    print(f"✓ Successfully generated {len(products)} products\n")
    
    # Display sample products
    print("Sample products:")
    for i, product in enumerate(products[:5], 1):
        print(f"  {i}. {product.get('name', 'Unknown')} - ${product.get('price', 0):.2f} ({product.get('profit_margin', 0):.1f}% margin)")
    
    if len(products) > 5:
        print(f"  ... and {len(products) - 5} more products")
    
except Exception as e:
    print(f"✗ Error generating products: {str(e)}")
    print(f"\nResponse preview: {result.products_json[:500]}...")
    products = []

# Save to file if products were generated
if products:
    data_folder = Path(__file__).parent / "datasets"
    data_folder.mkdir(exist_ok=True)
    output_file = data_folder / "skincare_products.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(products, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Saved to: {output_file}")
    
    # Display statistics
    avg_price = sum(p['price'] for p in products) / len(products)
    avg_margin = sum(p['profit_margin'] for p in products) / len(products)
    high_margin_count = sum(1 for p in products if p['profit_margin'] > 60)
    price_ranges = {
        'Budget (<$40)': sum(1 for p in products if p['price'] < 40),
        'Mid-range ($40-100)': sum(1 for p in products if 40 <= p['price'] <= 100),
        'High-end (>$100)': sum(1 for p in products if p['price'] > 100)
    }
    
    print(f"\nStatistics:")
    print(f"  Average Price: ${avg_price:.2f}")
    print(f"  Average Profit Margin: {avg_margin:.2f}%")
    print(f"  High Margin Products (>60%): {high_margin_count}")
    print(f"\n  Price Distribution:")
    for range_name, count in price_ranges.items():
        print(f"    {range_name}: {count} products")

# Inspect DSPy history (now just 1 call!)
print("\n" + "="*60)
print("DSPy History (Single API Call):")
print("="*60)
dspy.inspect_history(n=1)