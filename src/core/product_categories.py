"""
ABOUTME: Product Category Configuration System
ABOUTME: Provides optimized SSR parameters for different product types

Different product categories require different temperature settings and
amplification strengths for optimal rating differentiation.

Based on empirical testing:
- Luxury items: Need moderate temperature, strong amplification
- Budget items: Need sharper temperature
- Controversial: Need strong amplification
- Mainstream: Use default settings
"""

from typing import Dict, Optional
from dataclasses import dataclass
from enum import Enum


class ProductCategory(str, Enum):
    """Product category types with different characteristics"""

    # High-end luxury products ($1000+)
    LUXURY = "luxury"

    # Budget/value products (<$100)
    BUDGET = "budget"

    # Controversial or polarizing products
    CONTROVERSIAL = "controversial"

    # Age-specific products (gaming, anti-aging, etc.)
    AGE_SPECIFIC = "age_specific"

    # Gender-specific products
    GENDER_SPECIFIC = "gender_specific"

    # Ethical/values-based products (vegan, sustainable, etc.)
    ETHICAL = "ethical"

    # Mainstream consumer products (default)
    CONSUMER_GOODS = "consumer_goods"

    # Electronics and technology
    ELECTRONICS = "electronics"

    # Fashion and apparel
    FASHION = "fashion"

    # Food and beverages
    FOOD_BEVERAGE = "food_beverage"

    # Health and wellness
    HEALTH_WELLNESS = "health_wellness"

    # Home and garden
    HOME_GARDEN = "home_garden"


@dataclass
class CategoryConfig:
    """
    Configuration for a product category

    Attributes:
        category: Product category
        temperature: SSR temperature (lower = sharper distributions)
        amplification_strength: Sentiment amplification strength (0.0-1.0)
        amplification_enabled: Whether to use sentiment amplification
        min_confidence: Minimum confidence for amplification
        description: Category description
    """

    category: ProductCategory
    temperature: float
    amplification_strength: float
    amplification_enabled: bool = True
    min_confidence: float = 0.5
    description: str = ""


# Default category configurations (optimized from testing)
CATEGORY_CONFIGS: Dict[ProductCategory, CategoryConfig] = {
    ProductCategory.LUXURY: CategoryConfig(
        category=ProductCategory.LUXURY,
        temperature=0.4,
        amplification_strength=0.4,
        amplification_enabled=True,
        min_confidence=0.5,
        description="High-end luxury products with strong income effects"
    ),

    ProductCategory.BUDGET: CategoryConfig(
        category=ProductCategory.BUDGET,
        temperature=0.3,
        amplification_strength=0.3,
        amplification_enabled=True,
        min_confidence=0.5,
        description="Budget/value products with inverted income preferences"
    ),

    ProductCategory.CONTROVERSIAL: CategoryConfig(
        category=ProductCategory.CONTROVERSIAL,
        temperature=0.5,
        amplification_strength=0.5,
        amplification_enabled=True,
        min_confidence=0.4,
        description="Polarizing products with strong opinion variance"
    ),

    ProductCategory.AGE_SPECIFIC: CategoryConfig(
        category=ProductCategory.AGE_SPECIFIC,
        temperature=0.5,
        amplification_strength=0.4,
        amplification_enabled=True,
        min_confidence=0.5,
        description="Products targeting specific age groups"
    ),

    ProductCategory.GENDER_SPECIFIC: CategoryConfig(
        category=ProductCategory.GENDER_SPECIFIC,
        temperature=0.5,
        amplification_strength=0.3,
        amplification_enabled=True,
        min_confidence=0.5,
        description="Products with gender-specific appeal"
    ),

    ProductCategory.ETHICAL: CategoryConfig(
        category=ProductCategory.ETHICAL,
        temperature=0.5,
        amplification_strength=0.4,
        amplification_enabled=True,
        min_confidence=0.4,
        description="Ethical/values-based products (vegan, sustainable)"
    ),

    ProductCategory.CONSUMER_GOODS: CategoryConfig(
        category=ProductCategory.CONSUMER_GOODS,
        temperature=0.5,
        amplification_strength=0.3,
        amplification_enabled=True,
        min_confidence=0.5,
        description="Mainstream consumer products (default)"
    ),

    ProductCategory.ELECTRONICS: CategoryConfig(
        category=ProductCategory.ELECTRONICS,
        temperature=0.5,
        amplification_strength=0.3,
        amplification_enabled=True,
        min_confidence=0.5,
        description="Electronics and technology products"
    ),

    ProductCategory.FASHION: CategoryConfig(
        category=ProductCategory.FASHION,
        temperature=0.5,
        amplification_strength=0.3,
        amplification_enabled=True,
        min_confidence=0.5,
        description="Fashion and apparel products"
    ),

    ProductCategory.FOOD_BEVERAGE: CategoryConfig(
        category=ProductCategory.FOOD_BEVERAGE,
        temperature=0.4,
        amplification_strength=0.3,
        amplification_enabled=True,
        min_confidence=0.5,
        description="Food and beverage products"
    ),

    ProductCategory.HEALTH_WELLNESS: CategoryConfig(
        category=ProductCategory.HEALTH_WELLNESS,
        temperature=0.5,
        amplification_strength=0.3,
        amplification_enabled=True,
        min_confidence=0.5,
        description="Health and wellness products"
    ),

    ProductCategory.HOME_GARDEN: CategoryConfig(
        category=ProductCategory.HOME_GARDEN,
        temperature=0.5,
        amplification_strength=0.2,
        amplification_enabled=True,
        min_confidence=0.5,
        description="Home and garden products"
    ),
}


class ProductCategoryManager:
    """
    Manages product category configurations

    Provides category detection and optimal SSR configuration
    for different product types.
    """

    def __init__(self, custom_configs: Optional[Dict[ProductCategory, CategoryConfig]] = None):
        """
        Initialize category manager

        Args:
            custom_configs: Optional custom category configurations (overrides defaults)
        """
        self.configs = CATEGORY_CONFIGS.copy()
        if custom_configs:
            self.configs.update(custom_configs)

    def get_config(self, category: ProductCategory) -> CategoryConfig:
        """
        Get configuration for product category

        Args:
            category: Product category

        Returns:
            CategoryConfig for the category

        Raises:
            KeyError: If category not found
        """
        if category not in self.configs:
            # Return default consumer goods config
            return self.configs[ProductCategory.CONSUMER_GOODS]

        return self.configs[category]

    def detect_category(
        self,
        product_name: str,
        product_description: str,
        price: Optional[float] = None
    ) -> ProductCategory:
        """
        Auto-detect product category from name, description, and price

        Args:
            product_name: Product name
            product_description: Product description
            price: Optional price in USD

        Returns:
            Detected ProductCategory
        """
        text = (product_name + " " + product_description).lower()

        # Price-based detection
        if price is not None:
            if price >= 1000:
                return ProductCategory.LUXURY
            elif price < 100:
                return ProductCategory.BUDGET

        # Keyword-based detection
        category_keywords = {
            ProductCategory.CONTROVERSIAL: [
                'cricket', 'insect', 'vegan', 'plant-based', 'crypto',
                'nft', 'controversial', 'unconventional'
            ],
            ProductCategory.AGE_SPECIFIC: [
                'gaming', 'gamer', 'anti-aging', 'wrinkle', 'retinol',
                'senior', 'youth', 'teen', 'kids'
            ],
            ProductCategory.ETHICAL: [
                'vegan', 'cruelty-free', 'sustainable', 'eco-friendly',
                'organic', 'ethical', 'fair trade'
            ],
            ProductCategory.ELECTRONICS: [
                'smartphone', 'laptop', 'tablet', 'smartwatch', 'earbuds',
                'gaming', 'tech', 'electronic'
            ],
            ProductCategory.FASHION: [
                'jacket', 'dress', 'shirt', 'pants', 'shoes', 'accessory',
                'leather', 'fashion', 'apparel'
            ],
            ProductCategory.FOOD_BEVERAGE: [
                'food', 'beverage', 'drink', 'snack', 'protein bar',
                'coffee', 'tea', 'meal'
            ],
            ProductCategory.HEALTH_WELLNESS: [
                'serum', 'supplement', 'vitamin', 'fitness', 'wellness',
                'health', 'medical', 'skincare'
            ],
            ProductCategory.HOME_GARDEN: [
                'furniture', 'decor', 'garden', 'kitchen', 'appliance',
                'home', 'house', 'plant'
            ],
        }

        # Check keywords
        for category, keywords in category_keywords.items():
            if any(keyword in text for keyword in keywords):
                return category

        # Default to consumer goods
        return ProductCategory.CONSUMER_GOODS

    def get_config_for_product(
        self,
        product_name: str,
        product_description: str,
        price: Optional[float] = None,
        category_hint: Optional[ProductCategory] = None
    ) -> CategoryConfig:
        """
        Get optimal configuration for a product

        Args:
            product_name: Product name
            product_description: Product description
            price: Optional price in USD
            category_hint: Optional category hint (overrides auto-detection)

        Returns:
            CategoryConfig optimized for the product
        """
        if category_hint is not None:
            category = category_hint
        else:
            category = self.detect_category(product_name, product_description, price)

        return self.get_config(category)

    def list_categories(self) -> Dict[ProductCategory, CategoryConfig]:
        """Get all available category configurations"""
        return self.configs.copy()


# Singleton instance
_category_manager = ProductCategoryManager()


def get_category_manager() -> ProductCategoryManager:
    """Get global category manager instance"""
    return _category_manager


# Example usage and testing
if __name__ == "__main__":
    manager = ProductCategoryManager()

    print("Product Category Configuration System")
    print("=" * 70)

    # Test auto-detection
    test_products = [
        ("Patek Philippe Watch", "$45,000 luxury watch", 45000),
        ("EcoPhone Basic", "$79 budget smartphone", 79),
        ("Cricket Protein Bars", "protein bars made from crickets", 25),
        ("ProGamer Racing Chair", "gaming chair with RGB lighting", 449),
        ("Vegan Leather Jacket", "cruelty-free faux leather", 299),
    ]

    print("\nAuto-Detection Test:")
    print("-" * 70)

    for name, desc, price in test_products:
        category = manager.detect_category(name, desc, price)
        config = manager.get_config(category)

        print(f"\nProduct: {name} (${price})")
        print(f"  Detected Category: {category}")
        print(f"  Temperature: {config.temperature}")
        print(f"  Amplification: {config.amplification_strength}")
        print(f"  Description: {config.description}")

    # List all categories
    print("\n" + "=" * 70)
    print("All Category Configurations:")
    print("-" * 70)

    for category, config in manager.list_categories().items():
        print(f"\n{category.value:20} | T={config.temperature:.1f} | "
              f"Amp={config.amplification_strength:.1f}")
        print(f"  {config.description}")

    print("\n" + "=" * 70)
