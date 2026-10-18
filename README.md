# \# Instamart Customer Intelligence \& Support Copilot

# 

# \## Part A: Data audit (raw data)

# 

# | Check | Count |

# |---|---|

# | Order rows | 13,084 |

# | Exact duplicate rows | 119 |

# | Pending rows | 656 |

# | Distinct order\_id | 12,309 |

# | Non-numeric order\_amount | 120 |

# | Negative delivery\_minutes | 106 |

# | Blank delivery\_minutes | 2,541 |

# | Blank customer age | 61 |

# | Ages outside 10–100 | 6 |

# | Raw city spellings | 24 |

# | Raw gender spellings | 10 |

# | Blank gender | 30 |

# 

# Reproduce: `python audit.py`
## Part A: Cleaning rules

| Rule | Rows or values changed |
|---|---|
| Convert order_ts to datetime | 13,084 rows |
| Remove exact duplicate rows | 119 |
| Keep latest record per order_id | 656 (12,965 → 12,309) |
| Fill non-numeric or blank order_amount from price | 112 (after deduplication) |
| Negative delivery_minutes set to missing | TBD |
| Fill missing delivery_minutes (completed) with category median | TBD |
| Normalise city (strip, title case) | TBD |
| Map gender spellings to male/female/unknown | TBD |
| Ages outside 10–100 set to missing, then filled by city median | 67 missing before fill |

Reproduce: `python clean_customers.py`, `python clean_orders.py`, `python store.py`, `python validate.py`

Data: completed orders only, 11,110 of them, across six cities, from April to September 2025, totalling INR 2,998,668.80.
Distribution: mean 269.91 versus median 135.00, skew 4.61, quartiles 64.80 and 297.00.
Outliers: 897 values above 645.30, none below the lower fence, which is negative.
Relationships: the strongest pair is order_amount with discount_pct at -0.095, then order_amount with quantity at +0.087. Both are weak, and correlation doesn’t show cause.


### 1. Data Overview
The dataset contains 11,110 completed orders across six cities from April to September 2025, with a total order value of INR 2,998,668.80.

### 2. Distribution
The mean order amount (269.91) is higher than the median (135.00), with a skewness of 4.61, indicating a strongly right-skewed distribution. The first and third quartiles are 64.80 and 297.00, respectively.

### 3. Outliers
There are 897 outliers above the upper fence of 645.30 and no outliers below the lower fence, which is negative. This indicates that unusually high order amounts contribute to the right-skewed distribution.

### 4. Relationships
The strongest correlation is between order amount and discount percentage (-0.095), followed by order amount and quantity (+0.087). Both relationships are very weak, and correlation does not establish causation.

**Missing values:** The `.corr()` method handles missing values pairwise, so each correlation involving `delivery_minutes` uses only rows where both variables have valid values.
Rule	Rows or values changed
Negative delivery_minutes set to missing	105
Completed orders with missing delivery_minutes filled from category median	774 (includes 105 negatives set to missing)