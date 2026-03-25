# Hotel Profile Ingestion Requirement

## Objective
Ingest hotel profile information daily from the source system and identify new, modified, and removed properties.

## Business Rules
- A new record is one where property_id is not present in the target snapshot.
- A modified record is one where property_id exists but one or more tracked attributes changed.
- A removed record is one that was present in the previous snapshot but not in the latest source extract.
- Track address, phone, geo coordinates, brand, and property status.

## Data Quality
- property_id must not be null.
- latitude and longitude must be valid numbers.
- property_status must belong to allowed values ACTIVE, INACTIVE, CLOSED.

## Output Fields
- property_id
- change_type
- effective_date
- source_system
