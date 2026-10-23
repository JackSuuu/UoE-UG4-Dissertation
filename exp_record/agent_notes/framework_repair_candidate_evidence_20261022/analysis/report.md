# Repair-only candidate coverage

|Root|Slot|Valid|Complete success|Actions|Queries|Repair actions|
|---|---|---|---|---|---|---|
|1|0|True|True|444|444|0|
|1|1|True|True|444|444|0|
|0|0|True|False|520|523|96|
|0|1|True|False|520|523|96|
|0|2|True|False|520|523|96|
|0|3|True|False|520|523|96|
|4|0|True|False|520|523|80|
|4|1|True|False|520|523|96|
|4|2|True|False|520|523|85|
|4|3|True|True|506|509|11|

Candidate coverage: `[{"root": 0, "unique_first_actions": 1, "evaluated": 4, "successful_slots": []}, {"root": 4, "unique_first_actions": 4, "evaluated": 4, "successful_slots": [3]}]`

Selection was fixed-slot collection, not learned ranking. All outcomes are development evidence.
