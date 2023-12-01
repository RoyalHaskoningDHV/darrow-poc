# DARROW-POC
Documentation and code for onboarding a timeseries machine learning model to the `darrow-ml-platform`.

## Data models
The data is represented by a __rooted tree__ that mimics the real world (usually physical) relationships inherent in the data. This is easiest to understand with an example:

![data_model_example](images/image.png)

In this example, a tenant named Alice manages a wastewater treatment plant (WWTP) with two lines with various components and equipment, such as aeration blowers and pumps. For the rest of this document, we will name the nodes of the tree `units` and their respective measurements or other timeseries data `tags`.

This tree is not just represented in the figure, but also in the code used by the darrow platform. That is, there are specific `UNIT` objects, `TAG` objects and others that are used to define a rooted tree consisting of one parent node (tenant), with one or more children nodes (in this case WWTP), with one or more children nodes (in this case lines) etc.

When creating your own `ModelInterfaceV4` compliant model for onboarding onto the platform, keeping this structure in mind is very useful. Below we will go into more depth about how the `ModelInterfaceV4` relates to this rooted tree.

## `ModelInterfaceV4`: A contract between models and infrastructure
The `ModelInterfaceV4` is a python _Protocol_. That means, it specifies exactly what methods or attributes need to be defined, which parameters need to be inputted and what needs to be returned by a class in order to be a Protocol of type `ModelInterfaceV4`. Unlike a _Base class_, it does not allow for inheritance. Because of this it also does not allow an `__init__()` method. Think of it is as a recipe to follow.

To verify if a class follows the contract an `isinstance` check of the form `isinstance(myclass, myprotocol)` can be performed. For `ModelInterfaceV4`, we are using a modified version of `Protocol`, called `AnnotationProtocol` from the [`annotation-protocol` package](https://github.com/RoyalHaskoningDHV/annotation-protocol). It allows for more thorough `isinstance` checks, making all the necessary comparisons. An example implementation for how to test `ModelInterfaceV4` compliance can be found in `tests/test_inteface.py`.

## Proof of concept / example model
This repository contains an example model called `POCAnomaly` (in `models/poc.py`) adhering to `ModelInterfaceV4`. It is an anomaly detection model that takes sensor data as input and replaces anomalies with `np.nan`. Of course, the purpose of the machine learning models here is not important. Instead, we aim to show how a machine learning model can be onboarded onto the `darrow-ml-platform` (or at least prepared for onboarding by complying to the data contract).

We also included a local _Executor_ of the model in `mocks.mocks.py`, which mimicks how a real executor would execute model training or predicting on the `darrow-ml-platform` infrastructure. It is a lot simpler, but should give a decent idea of what happens to the model during execution of either training or predicting.

## Walkthrough
In `notebooks/walkthrough.ipynb` you can find a step-by-step guide of how to make a model `ModelInterfaceV4` compliant.

## How `ModelInterfaceV4` relates to the __rooted tree__
While the `ModelInterfaceV4` protocol is not terribly complex, it contains a number of custom `types` and `Enums`, which often relate to the __rooted tree__ data model and might take some getting used to. Consider the `Node` object from the `objectmodels.hierarchy` module:

```python
@dataclass
class Node:
    val: Unit
    parent: Node | None = None
    children: list[Node] | None = None
```

It has a value, which is of type `Unit`, a parent, which is also a `Node` (or `None`), and children Nodes. So we can build a tree out of nodes. Let's investigate the `Unit` class:

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
```

In practice, you do not have to define any tree structure yourself. We should come up with a definition together with the consortium partners and then RHDHV will implement the tree and make it available to the partners. That being said it is still useful to have an idea about these classes, since we have to define a number of related ones when onboarding our model to `ModelInterfaceV4`. Let's look at that next.

## Deep dive into `ModelInterfaceV4`
Let's have a look at what we need to define to make our model `ModelInterfaceV4` compliant. For illustration purposes let's just use the proof of concept model from this repository. We want to onboard an anomaly detection model, defined in `models.anomaly_detection`. But before we get to that we should have a quick look at our data.

### Data
The data for this proof of concept looks as follows, where `altenburg1` is one of the sensors, which measures water `discharge`. There are multiple of these sensors, plus two rainfall sensors of type `precip` and one evaporation sensor of type `evap`. Each datapoint needs to have both `ID` and `TYPE` specified, even if all sensors have the same `TYPE`.

| TIME                      | ID         | VALUE    | TYPE      |
|---------------------------|------------|----------|-----------|
| 2018-01-01 00:00:00+00:00 | altenburg1 | 33.22525 | discharge |
| 2018-01-01 01:00:00+00:00 | altenburg1 | 5234     | discharge |

The data above is following the `SAM` long-format, which is also how the data is currently saved in `tests/testing_data/*.parquet`. In practice, the data will be connected via an `ADX` store and be read in following the `InputData` format. In that format each sensor is read in separately as a `pandas.DataFrame`. If you are curious about this format you can checkout the [Walkthrough](#walkthrough).

### 

```python

```

```python
class StahModel(ModelInterfaceV4):

    model_type_name: str = "afvoervoorspellingen_stah"
    # Model category is based on the output of the model.
    model_category: ModelCategory = ModelCategory.prediction
    # Number between (-inf, inf) indicating the model performance.
    performance_value: float
    # List of features used to train the model. If not supplied, equal to data_config().
    train_data_config: dict[DataLevels, list] | None = None
    # This is only needed when get_target_tag_template returns UnitTagTemplate
    target: UnitTag | None = None
```
