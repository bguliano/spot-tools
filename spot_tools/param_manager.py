import logging
from dataclasses import dataclass
from typing import Optional, Union, Iterable, Type, Dict, Any

from bosdyn.api import service_customization_pb2
from bosdyn.api.units_pb2 import Units
from bosdyn.client.service_customization_helpers import validate_dict_spec, InvalidCustomParamSpecError
from google.protobuf.wrappers_pb2 import Int64Value, BoolValue


@dataclass
class ParamInfoExt:
    key: str
    display_name: str
    description: str = ''
    display_order: Optional[int] = None


@dataclass
class IntParam(ParamInfoExt):
    default_value: Optional[int] = None
    units: Optional[str] = None
    min_value: Optional[int] = None
    max_value: Optional[int] = None


@dataclass
class DoubleParam(ParamInfoExt):
    default_value: Optional[int] = None
    units: Optional[str] = None
    min_value: Optional[int] = None
    max_value: Optional[int] = None


@dataclass
class StringParam(ParamInfoExt):
    options: Optional[Iterable[str]] = None
    editable: Optional[bool] = None
    default_value: Optional[str] = None


@dataclass
class BoolParam(ParamInfoExt):
    default_value: Optional[bool] = None


def translate_spec_type(spec) -> str:
    if isinstance(spec, service_customization_pb2.DictParam.Spec):
        return 'dict_spec'
    # if isinstance(spec, service_customization_pb2.ListParam.Spec):
    #     return 'list_spec'
    if isinstance(spec, service_customization_pb2.Int64Param.Spec):
        return 'int_spec'
    if isinstance(spec, service_customization_pb2.DoubleParam.Spec):
        return 'double_spec'
    if isinstance(spec, service_customization_pb2.StringParam.Spec):
        return 'string_spec'
    # if isinstance(spec, service_customization_pb2.RegionOfInterestParam.Spec):
    #     return 'roi_spec'
    if isinstance(spec, service_customization_pb2.BoolParam.Spec):
        return 'bool_spec'
    if isinstance(spec, service_customization_pb2.OneOfParam.Spec):
        return 'one_of_spec'
    else:
        raise ValueError(f'Spec of type {type(spec)} is not supported yet')


def translate_param_type(param_type) -> str:
    if param_type == DictParam:
        return 'dict_value'
    # if param_type == ListParam:
    #     return 'list_spec'
    if param_type == IntParam:
        return 'int_value'
    if param_type == DoubleParam:
        return 'double_value'
    if param_type == StringParam:
        return 'string_value'
    # if param_type == ROIParam:
    #     return 'roi_spec'
    if param_type == BoolParam:
        return 'bool_value'
    if param_type == OneOfParam:
        return 'one_of_value'
    else:
        raise ValueError(f'Param of type {type(param_type)} is not supported yet')


def load_specs(params: Iterable, *, onto_spec) -> Dict[str, Type]:
    found_keys_to_param_type = {}

    for param in params:
        found_keys_to_param_type[param.key] = type(param)

        # create spec objects
        if isinstance(param, Union[DictParam, OneOfParam]):
            spec = param.spec
            found_keys_to_param_type.update(param.keys_to_param_type)

        elif isinstance(param, IntParam):
            spec = service_customization_pb2.Int64Param.Spec()
            if param.default_value is not None:
                spec.default_value.CopyFrom(Int64Value(value=param.default_value))
            if param.units is not None:
                spec.units.CopyFrom(Units(name=param.units))
            if param.min_value is not None:
                spec.min_value.CopyFrom(Int64Value(value=param.min_value))
            if param.max_value is not None:
                spec.max_value.CopyFrom(Int64Value(value=param.max_value))

        elif isinstance(param, DoubleParam):
            spec = service_customization_pb2.DoubleParam.Spec()
            if param.default_value is not None:
                spec.default_value.CopyFrom(Int64Value(value=param.default_value))
            if param.units is not None:
                spec.units.CopyFrom(Units(name=param.units))
            if param.min_value is not None:
                spec.min_value.CopyFrom(Int64Value(value=param.min_value))
            if param.max_value is not None:
                spec.max_value.CopyFrom(Int64Value(value=param.max_value))

        elif isinstance(param, StringParam):
            spec = service_customization_pb2.StringParam.Spec()
            if param.options is not None:
                spec.options.extend(param.options)
            if param.editable is not None:
                spec.editable = param.editable
            if param.default_value is not None:
                spec.default_value = param.default_value

        elif isinstance(param, BoolParam):
            spec = service_customization_pb2.BoolParam.Spec()
            if param.default_value is not None:
                spec.default_value.CopyFrom(BoolValue(value=param.default_value))

        else:
            raise ValueError(f'Param of type {type(param)} is not supported yet')

        # create ui info to go along with these spec objects
        ui_info = service_customization_pb2.UserInterfaceInfo()
        ui_info.display_name = param.display_name
        ui_info.description = param.description
        if param.display_order is not None:
            ui_info.display_order = param.display_order

        # copy spec into container spec
        if isinstance(onto_spec, service_customization_pb2.OneOfParam.Spec):
            onto_spec.specs[param.key].spec.CopyFrom(spec)
        elif isinstance(onto_spec, service_customization_pb2.DictParam.Spec):
            # translate spec to container spec name
            custom_param_spec_field = translate_spec_type(spec)
            getattr(onto_spec.specs[param.key].spec, custom_param_spec_field).CopyFrom(spec)
        else:
            raise ValueError(f'Spec of type {type(spec)} is not supported yet')
        # copy matching ui info into container spec
        onto_spec.specs[param.key].ui_info.CopyFrom(ui_info)

    return found_keys_to_param_type


class DictParam(ParamInfoExt):
    def __init__(self, *args, key: str, display_name: str, description: str = '', display_order: Optional[int] = None,
                 is_hidden_by_default: Optional[bool] = None):
        super().__init__(key, display_name, description, display_order)

        self.spec = service_customization_pb2.DictParam.Spec()
        if is_hidden_by_default is not None:
            self.spec.is_hidden_by_default = is_hidden_by_default

        self.keys_to_param_type = load_specs(args, onto_spec=self.spec)


class OneOfParam(ParamInfoExt):
    def __init__(self, *args, key: str, display_name: str, description: str = '', display_order: Optional[int] = None,
                 default_key: Optional[str] = None):
        super().__init__(key, display_name, description, display_order)

        self.spec = service_customization_pb2.OneOfParam.Spec()
        if default_key is not None:
            self.spec.default_key = default_key

        self.keys_to_param_type = load_specs(args, onto_spec=self.spec)


class ParamManager:
    def __init__(self, *, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.params = service_customization_pb2.DictParam.Spec()
        self.keys_to_param_type: Dict[str, Type] = {}

    def add_params(self, *args):
        all_keys = load_specs(args, onto_spec=self.params)
        self.keys_to_param_type.update(all_keys)
        self._validate()

    def parse_request_params(self, request) -> Dict[str, Any]:

        # recursively parse every param, looking for their values
        def scan_container_param(param) -> Dict[str, Any]:
            results = {}
            custom_param_dict = param.values
            for child_key, child_param in custom_param_dict.items():
                custom_param_type = self.keys_to_param_type[child_key]
                # child_param is a CustomParam if from DictParam, otherwise normal
                if isinstance(param, service_customization_pb2.DictParam):
                    correct_param = getattr(child_param, translate_param_type(custom_param_type))
                elif isinstance(param, service_customization_pb2.OneOfParam):
                    correct_param = child_param
                else:
                    raise ValueError(f'A param of type {custom_param_type} should not be here')
                # anything below this comment referencing child_param should reference correct_param instead
                if hasattr(correct_param, 'value'):
                    results[child_key] = correct_param.value
                else:  # assume it's a container
                    results.update(scan_container_param(correct_param))
                if isinstance(correct_param, service_customization_pb2.OneOfParam):
                    # this is a weird case, which is why it's down here...OneOfParams are
                    # containers AND they have a value that needs to be recorded
                    results[child_key] = correct_param.key
            return results

        return scan_container_param(request.params)

    def _validate(self):
        try:
            validate_dict_spec(self.params)
        except InvalidCustomParamSpecError:
            self.logger.exception(f'Custom params failed validation')
            # clear the custom parameters if they are invalid
            self.params.Clear()
            self.keys_to_param_type.clear()
