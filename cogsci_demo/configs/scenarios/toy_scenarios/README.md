# Toy Scenarios for Agent Action Testing

This directory contains toy scenarios designed to test individual actions of the greedy agent. Each scenario is carefully crafted to trigger a specific action while ensuring no stockouts occur (which would result in a -1000000 penalty).

## Scenario Design Principles

- **18 stable drugs + 1 test drug**: Each scenario uses 18 drugs from `example_drugs.csv` with stable configurations, plus one drug specifically designed to test the target action
- **10-week duration**: All scenarios are designed to run for 10 weeks
- **No stockouts**: All scenarios are designed to avoid stockouts by ensuring the greedy agent takes appropriate actions
- **Single action focus**: Each scenario tests exactly one action type
- **Moderate challenge**: Scenarios are designed to be moderately challenging, not crisis-level, to test agent logic without creating panic situations

## Available Scenarios

### 1. `test_audit_action.json`
**Target Action**: `AUDIT_QOH_AND_UTZ`
**Trigger Conditions**: 
- Low runway (< 4 weeks)
- High uncertainty (erd_uncertainty > 1.5)
**Test Drug**: Hydromorphone 2mg/1ml SDV with 2 weeks runway and 2.0 uncertainty

### 2. `test_soft_lma_action.json`
**Target Action**: `IMPLEMENT_SOFT_LMA`
**Trigger Conditions**: 
- Medium runway (4-6 weeks)
**Test Drug**: Hydromorphone 2mg/1ml SDV with 4 weeks runway

### 3. `test_hard_lma_action.json`
**Target Action**: `IMPLEMENT_HARD_LMA`
**Trigger Conditions**: 
- Low runway (< 4 weeks)
**Test Drug**: Hydromorphone 2mg/1ml SDV with 2 weeks runway

### 4. `test_query_supplier_action.json`
**Target Action**: `QUERY_ALTERNATE_SUPPLIER`
**Trigger Conditions**: 
- Low runway (< 4 weeks)
- High ERD uncertainty (> 2.0)
- High ERD weeks (> 4.0)
**Test Drug**: Hydromorphone 2mg/1ml SDV with 2 weeks runway, 2.5 uncertainty, and 5.0 ERD

### 5. `test_switch_supplier_action.json`
**Target Action**: `SWITCH_SUPPLIER`
**Trigger Conditions**: 
- Low runway (< 2 weeks)
- Unreliable primary supplier (< 0.7)
- High ERD (> 6.0)
**Test Drug**: Hydromorphone 2mg/1ml SDV with 1 week runway, 0.6 primary reliability, and 7.0 ERD

### 6. `test_switch_back_action.json`
**Target Action**: `SWITCH_BACK_TO_PRIMARY`
**Trigger Conditions**: 
- Currently using alternate supplier
- Alternate supplier is unreliable (< 0.6)
- High alternate ERD (> 8.0)
**Test Drug**: Hydromorphone 2mg/1ml SDV using alternate supplier with 0.5 reliability and 9.0 ERD

### 7. `test_reserve_warehouse_action.json`
**Target Action**: `REQUEST_FROM_RESERVE_WAREHOUSE`
**Trigger Conditions**: 
- Low runway (< 2 weeks)
**Test Drug**: Hydromorphone 2mg/1ml SDV with 1 week runway

### 8. `test_loan_hospitals_action.json`
**Target Action**: `REQUEST_LOAN_FROM_OTHER_HOSPITALS`
**Trigger Conditions**: 
- Low runway (< 2 weeks)
**Test Drug**: Hydromorphone 2mg/1ml SDV with 1 week runway

### 9. `test_grey_market_action.json`
**Target Action**: `GREY_MARKET_PURCHASE`
**Trigger Conditions**: 
- Low runway (<= 1 week)
**Test Drug**: Hydromorphone 2mg/1ml SDV with 1 week runway

### 10. `test_contact_manufacturer_action.json`
**Target Action**: `CONTACT_MANUFACTURER_DIRECT`
**Trigger Conditions**: 
- Low runway (< 4 weeks)
- High ERD (> 4.0)
**Test Drug**: Hydromorphone 2mg/1ml SDV with 2 weeks runway and 8.0 ERD

## Usage

These scenarios can be used to verify that the greedy agent logic works correctly by:

1. Running each scenario with the greedy agent
2. Verifying that the expected action is taken for the test drug
3. Confirming that no stockouts occur (rewards should not be extremely negative)
4. Checking that the agent's decision-making logic is sound

## Expected Outcomes

When running these scenarios with the greedy agent:
- The test drug should trigger the expected action
- All other drugs should remain stable
- No stockouts should occur
- Rewards should be reasonable (not hitting the -1000000 stockout penalty)
- The agent should demonstrate proper decision-making logic

This provides a systematic way to validate that each component of the greedy agent's logic works as intended.
