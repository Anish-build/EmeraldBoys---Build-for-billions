# Jal-Setu Database API

The database uses SQLite and is accessed through functions in `database/queries.py`.

## Pump

### create_pump()
Creates a new pump.

```python
create_pump(
    pump_id,
    location,
    pump_type=None,
    operator=None
)