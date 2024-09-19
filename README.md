# DataPlan

DataPlan provides a user-friendly API to pyarrow's Acero for performing streamed queries over data.

Note that Acero is powerful but not intelligent; care must be taken when constructing plans (e.g., select columns and apply filters early when possible).

