"""Actions Definitions"""

from enum import Enum, auto

class Action(Enum):
    """All Actions"""
    
    WAIT_MONITOR = auto()
    AUDIT_QOH_AND_UTZ = auto()
    IMPLEMENT_SOFT_LMA = auto()
    IMPLEMENT_HARD_LMA = auto()
    REMOVE_LMA = auto()
    QUERY_ALTERNATE_SUPPLIER = auto()
    CONTACT_MANUFACTURER_DIRECT = auto()
    SWITCH_SUPPLIER = auto()
    SWITCH_BACK_TO_PRIMARY = auto()
    REQUEST_FROM_RESERVE_WAREHOUSE = auto()
    REQUEST_LOAN_FROM_OTHER_HOSPITALS = auto()
    GREY_MARKET_PURCHASE = auto()
    
def all_actions():
    """All Actions Getter"""
    return [Action.WAIT_MONITOR, Action.AUDIT_QOH_AND_UTZ, Action.IMPLEMENT_SOFT_LMA, Action.IMPLEMENT_HARD_LMA, Action.REMOVE_LMA, Action.QUERY_ALTERNATE_SUPPLIER, Action.CONTACT_MANUFACTURER_DIRECT, Action.SWITCH_SUPPLIER, Action.SWITCH_BACK_TO_PRIMARY, Action.REQUEST_FROM_RESERVE_WAREHOUSE, Action.REQUEST_LOAN_FROM_OTHER_HOSPITALS, Action.GREY_MARKET_PURCHASE]
