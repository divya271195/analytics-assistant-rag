# Daily Booking Revenue Requirement

## Objective
Build a daily pipeline that calculates total paid booking revenue by hotel and market.

## Business Rules
- Only completed bookings should be counted.
- Cancelled bookings should be excluded.
- Revenue should be aggregated by booking_date, hotel_id, and market_id.
- Output should be partitioned by booking_date.

## Source Tables
- bookings
- hotel_dim
- market_dim

## Output Fields
- booking_date
- hotel_id
- market_id
- total_paid_revenue
- booking_count
