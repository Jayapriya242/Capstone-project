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

