# Changelog

## Version 0.1.8
- Remove hardcoded train date

## Version 0.1.7
- add linting and testing github actions pipeline `validate_pipeline.yml`.
- bugfix: Add datetime index to predictions dataframe.

## Version 0.1.6
- Added github action pipelines for publishing to `testpypi` and `pypi`.
- Remove old publishing pipeline to `SAM` artifacts.

## Version 0.1.5
- Moved mocks to [twinn-ml-interface](https://github.com/RoyalHaskoningDHV/twinn-ml-interface)

## Version 0.1.4
- Adapted to the last changes in the model interface

## Version 0.1.3
- Adjust to new configuration in myaquasuite.
- Make logging to MLflow a little nicer and avoid errors due to cut-off column names.

## Version 0.1.2
- Adapt to `twinn-ml-interface v0.2.7`.
- Only allow returning `list[DataLabelConfigTemplate]` in interface method `get_data_config_template`.

## Version 0.1.1
- Implement changes from `ModelInterfaceV4` Version 0.2.4 (add `Configuration` and `MetaDataLogger` to `load` method)

## Version 0.1.0
- Initial version of DARROW-POC
