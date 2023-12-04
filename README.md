# DARROW-POC
Documentation and code for onboarding a timeseries machine learning model to the `darrow-ml-platform`. You can use the example model and mock executors as a starting point for onboarding your own models (in principle the mock executor is not needed for this, but it gives a general idea of how the model is executed for training or predicting in the infrastructure).

## Hierarchy data model
The data hierarchy is represented by a __rooted tree__ that mimics the real world (usually physical) relationships inherent in the data. This is easiest to understand with an example:

![data_model_example](images/rooted_tree_docs.png)

In this example, a tenant named Alice manages a wastewater treatment plant (WWTP) with two lines with various components and equipment, such as aeration blowers and pumps.

This tree is not just represented in the figure, but also in the code used by the darrow platform. That is, there are specific `UNIT` objects, `TAG` objects and others that are used to define a rooted tree consisting of one parent node (tenant), with one or more children nodes (in this case WWTP), which can again have one or more children nodes (in this case lines) etc.

You will not need to create this tree yourself, this should be decided together with the consortium partners. Nevertheless, when creating your own `ModelInterfaceV4` compliant model for onboarding onto the platform, keeping this structure in mind is useful. Below we will go into more depth about how the `ModelInterfaceV4` relates to this rooted tree.

## `ModelInterfaceV4`: A contract between models and infrastructure
The `ModelInterfaceV4` is a python _Protocol_. That means, it specifies exactly what methods or attributes need to be defined, which parameters need to be inputted and what needs to be returned by methods of a class in order for the class to be a Protocol of type `ModelInterfaceV4`. Unlike a _Base class_, it does not allow for inheritance. Because of this it also does not have an `__init__()` method. You can think of it as a recipe to follow.

To verify if a class follows the contract an `isinstance` check of the form `isinstance(myclass, myprotocol)` can be performed. For `ModelInterfaceV4`, we are using a modified version of `Protocol`, called `AnnotationProtocol` from the [`annotation-protocol` package](https://github.com/RoyalHaskoningDHV/annotation-protocol). It allows for more thorough `isinstance` checks, making all the necessary comparisons. An example implementation for how to test `ModelInterfaceV4` compliance can be found in `tests/test_inteface.py`.

## Proof of concept / example model
This repository contains an example model called `POCAnomaly` (in `models/poc.py`) adhering to `ModelInterfaceV4`. It is an anomaly detection model that takes sensor data as input and replaces anomalies with `np.nan`. Of course, the purpose of the machine learning model here is not important. Instead, we aim to show how a machine learning model can be onboarded onto the `darrow-ml-platform` (or at least prepared for onboarding by complying to the data contract).

We also included a local _Executor_ of the model in `mocks/mocks.py`, which mimicks how a real executor would execute model training or predicting on the `darrow-ml-platform` infrastructure. It is a lot simpler, but should give a decent idea of what happens to the model during execution of either training or predicting.

## How `ModelInterfaceV4` relates to the __rooted tree__
While the `ModelInterfaceV4` protocol is not terribly complex, it contains a number of custom `types` and `Enums`, which often relate to the __rooted tree__ data model and might take some getting used to. Consider the `Node` object from the `objectmodels.hierarchy` module:

```python
@dataclass
class Node:
    val: Unit
    parent: Node | None = None
    children: list[Node] | None = None
```

It has a value, which is of type `Unit`, a parent, which is also a `Node` (or `None`), and children Nodes. These nodes are the building blocks for the __rooted_tree__. Let's investigate the `Unit` class:

```python
@dataclass
class Unit:
    unit_code: str
    unit_type_code: str
    active: bool
    name: str | None = None
    unit_type_name: str | None = None
    geometry: dict[str, list[float]] | None = None
    properties: list | None = None
    metadata: dict | None = None

    def __hash__(self):
        return hash(self.unit_code)

    def __eq__(self, other):
        if isinstance(other, Unit):
            return self.unit_code == other.unit_code
        return NotImplemented
```

This looks a bit more complicated, but it basically contains all the information about a given `Node` or `Unit` that we might have, besides the actual timeseries data. A simple unit definition could look like this:

```python
Unit(
    unit_code="stah",
    unit_type_code="DISCHARGE_STATION",
    active=True,
)
```

While you do not have to define any tree structure yourself it is still useful to have an idea about these classes, since we have to define a number of related ones when onboarding our model to `ModelInterfaceV4`. Let's look at that next.

## Deep dive into `ModelInterfaceV4`
Let's have a look at what we need to define to make our model `ModelInterfaceV4` compliant. For illustration purposes let's just use the proof of concept model from this repository. We want to onboard an anomaly detection model, defined in `models.anomaly_detection`. But before we get to that we should have a quick look at our data.

### POC Hierarchy
This is an illustration of our __rooted_tree__ hierarchy. In this case we do not have any meta-data and we have a single timeseries per measurement station.

![data_model_poc](images/rooted_tree_poc.png)

### Data
The data for this proof of concept looks as follows, where `altenburg1` is one of the station names (`unit_code`), which has a sensor that measures water `discharge` (`tag`). There are multiple of these stations, plus two rainfall stations with sensors of type `precipitation` and one evaporation station with a sensor of type `evaporation`. Each datapoint needs to have both `ID` and `TYPE` specified, even if all sensors have the same `TYPE`. Here, stations end up being the `ID` and sensor types end up being the `TYPE`.

| TIME                      | ID         | VALUE    | TYPE      |
|---------------------------|------------|----------|-----------|
| 2018-01-01 00:00:00+00:00 | altenburg1 | 33.22525 | discharge |
| 2018-01-01 01:00:00+00:00 | altenburg1 | 5234     | discharge |

The data above is following the `SAM` long-format, which is also how the data is currently saved in `tests/testing_data/*.parquet`. In practice, the data will be connected via an `ADX` store and be read in following the `InputData` format. In that format each sensor is read in separately as a `pandas.DataFrame`. If you are curious about this format you can checkout the class in `twinn-ml-interface/input_data/input_data.py`. Here, we create this data format from `SAM` long format in the mock executor (`mocks/mocks.py`).

### Implementing methods of `ModelInterfaceV4`

#### Methods you need to write an implementation for

All methods in `ModelInterfaceV4` need to be present for our model class to be compliant with the data contract. However, if we do not want to use a particular method, we can have it do nothing. Some methods, however do have to be implemented and do something - let's look at those first.

```python
@staticmethod
def get_target_template() -> UnitTagTemplate | UnitTag:
    return UnitTag(Unit("STAH", "DISCHARGE_STATION", True), Tag("DISCHARGE"))
```

The method `get_target_template()` returns either a `UnitTagTemplate` or a `UnitTag` and thereby specifies the _target_ variable of our machine learning model. The latter is somewhat simpler and used in this example. We basically specify the connection between our target unit, which has `unit_code='STAH'` and `unit_type_code='DISCHARGE_STATION'`, and the timeseries or sensor data we would like to get, which is given by the `tag` `"DISCHARGE"`. The unit information can be seen also in the rooted tree above. The tag label refers to the `TYPE` used in the data (TODO: Is that true?).

Next let's look at the `get_data_config_template()` method, which determines what data to select for our machine learning model besides the target.

```python
@staticmethod
def get_data_config_template() -> list[DataLabelConfigTemplate] | list[UnitTag]:
    return [
        UnitTag.from_string("altenburg1:discharge"),
        UnitTag.from_string("eschweiler:discharge"),
        UnitTag.from_string("herzogenrath1:discharge"),
        UnitTag.from_string("juelich:discharge"),
        UnitTag.from_string("stah:discharge"),
        UnitTag.from_string("evap:evaporation"),
        UnitTag.from_string("middenroer:precipitation"),
        UnitTag.from_string("urft:precipitation"),
    ]
```

There are again two possible implementations, either with `list[DataLabelConfigTemplate]` or `list[UnitTag]`. For illustration purposes we will show both. `UnitTag`, as used in the implementation above, we have already seen in the previous method. However, there is an alternative way to implement it, using the `from_string` class method, where we specify only the `unit_tag`, which is a combination between `unit_code` and `tag`, separated by a colon: `"{unit_code}:{tag}"`.

In the below implementation we use `DataLabelConfigTemplate` instead of `UnitTag`. This implementation is more complex, but takes advantage of relative paths in our __rooted tree__. The first `DataLabelConfigTemplate` selects all units following the path `RelativeType.CHILDREN` starting from the _target_ unit. TODO: ... I actually don't really know how this works properly...

```python
@staticmethod
def get_data_config_template() -> list[DataLabelConfigTemplate] | list[UnitTag]:
    return [
        DataLabelConfigTemplate(
            data_level=DataLevel.SENSOR,
            unit_tag_templates=[UnitTagTemplate([RelativeType.CHILDREN], [Tag("DISCHARGE")])],
            availability_level=AvailabilityLevel.available_until_now,
        ),
        DataLabelConfigTemplate(
            data_level=DataLevel.WEATHER,
            unit_tag_templates=[
                UnitTagTemplate([RelativeType.CHILDREN], [Tag("PRECIPITATION"), Tag("EVAPORATION")])
            ],
            availability_level=AvailabilityLevel.available_until_now,
        ),
    ]
```

The next method to implement should now be relatively straightforward, it is the `UnitTag` or `UnitTagTemplate` for the model output/results.

```python
@staticmethod
def get_result_template() -> UnitTagTemplate | UnitTag:
    return UnitTag(Unit("STAHROER", "DISCHARGE_STATION", True), Tag("DISCHARGE_FORECAST"))
```

The next method is our way to initialize certain attributes and what would have been in the `__init__()`, if a __Protocol__ had one. We initialize the model and the corresponding `MetaDataLogger`. We advise you to re-use this code and log things with the logger object in other methods you implement. The `Configuration` is not something you need to worry about, since it will be taken care of by the infrastructure. (TODO: Is that true?)

```python
@classmethod
def initialize(cls, configuration: Configuration, logger: MetaDataLogger) -> ModelInterfaceV4:
    model = cls(configuration.target_name)
    model.configuration = configuration
    model.logger = logger
    return model
```

The `train` method should implement the model training. Here, we first combine the input data into a `pandas.DataFrame` in wide format. Next, we initialize our machine learning model, fit and evaluate it (which includes splitting into train and validation sets), and finally log some parameters to the `MetaDataLogger`. These would later be logged to mlflow by the infrastructure if the model were to run non-locally. The trained model is saved as a hidden attribute in the `POCAnomaly` class.

```python
def train(self, input_data: InputData, **kwargs) -> None:
    train = pd.concat(input_data.values(), axis=1)
    validator = ValidationModel(
        train,
        model_type="lasso",
        n_features=5,
        use_precipitation_features=False,
        training_end_date="2010-01-04 00:00:00",
    )
    _, num_obs, _, r2 = validator.fit_and_evaluate()
    self.logger.log_params(validator.flatten_output(r2, "r2"))  # This will be logged to mlflow
    self.logger.log_params({f"samples_{k}": v for k, v in num_obs.items()})

    self._model = validator
```

Conversely, the `predict` method needs to implement the making of predictions with the trained model. In our case, this means replacing detected anomalies with `np.nan`. Note that with the tiny amount of sample training data saved in this repository, the quality of these predictions will always be abysmal.

```python
def predict(self, input_data: InputData, **kwargs) -> list[pd.DataFrame]:
    model = self._model
    X = pd.concat(input_data.values(), axis=1)
    X_removed_anomalies = model.predict(X)

    return X_removed_anomalies
```

Finally, the `dump` and `load` methods need to be defined to be able to save and re-load the model. In this example, we simply dump to and reload from `pickle`.

```python
def dump(self, foldername: PathLike, filename: str) -> None:
    with open(Path(foldername) / (filename + ".pkl"), "wb") as f:
        pickle.dump(self, f)
    return None

@classmethod
def load(cls, foldername: PathLike, filename: str) -> Callable:
    with open(Path(foldername) / (filename + ".pkl"), "rb") as f:
        model = pickle.load(f)
    return model
```

#### Methods you need to have, but do not need to implement

In principle, you can use the exact implementations given below if you do not want to implement them. If you do, `preprocess` and `validate_input_data` should be fairly self-explanatory. `get_train_window_finder_config_template` is useful when you want to tweak how a training window is chosen when (re-)training your model in production. This configuration selects the data used for determining that window. TODO: explain with an example

```python
@staticmethod
def get_train_window_finder_config_template() -> list[DataLabelConfigTemplate] | None:
    return None

def preprocess(self, input_data: InputData) -> InputData:
    return input_data

def validate_input_data(
    self,
    input_data: InputData,
) -> WindowViability:
    return True, "Input data is valid."
```

### Model attributes

```python
class POCAnomaly(ModelInterfaceV4):

    model_type_name: str = "pocanomaly"
    # Model category is based on the output of the model.
    model_category: ModelCategory = ModelCategory.ANOMALY
    # List of features used to train the model. If not supplied, equal to data_config().
    train_data_config: dict[DataLevels, list] | None = None
    # This is only needed when get_target_tag_template returns UnitTagTemplate
    target: UnitTag | None = None
```

`model_type_name` is simply the name of our model and can be any string (TODO is that true?). `model_category` has to be one of the possible levels of the `ModelCategory` Enum. The last two entries are optional and we assign default values as `None`.

### Testing compliance with the data contract

Since all the attributes and methods from the __Protocol__ `ModelInterfaceV4` are implemented, our `POCAnomaly` class passes the `isinstance` check with `ModelInterfaceV4` (see `tests/test_interface.py`).

## Mock Executor

The `executor` class takes care of running the model either for training or predictions on the `darrow-ml-platform` infrastructure. Here, we implemented a mock executor to emulate that behaviour to some extend, which hopefully makes it a little clearer in what context the model class (in this case `POCAnomaly`) will be used. In addition, you can run additional tests with this executor (`tests/test_executor.py`) that actually perform training and predicting.
