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
