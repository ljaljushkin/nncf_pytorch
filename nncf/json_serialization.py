import inspect
import json


class SerializableClassesRegistry:
    def __init__(self):
        self._serializable_classes = {}
        self._serializable_class_names = {}

    def register(self, name=None):
        def decorator(cls):
            class_name = name if name is not None else cls.__name__

            if class_name in self._serializable_class_names:
                raise ValueError(
                    '%s has already been registered to %s'.format(class_name, self._serializable_classes[class_name]))

            if cls in self._serializable_class_names:
                raise ValueError('%s has already been registered to %s' %
                                 (class_name, self._serializable_class_names[cls]))

            if inspect.isclass(cls) and not hasattr(cls, 'to_dict'):
                raise ValueError('Cannot register a class ({}) that does not have to_dict() method.'.format(class_name))

            self._serializable_class_names[cls] = class_name
            self._serializable_classes[class_name] = cls

            return cls

        return decorator

    def get_registered_class(self, class_name):
        result = None
        if class_name in self._serializable_classes:
            result = self._serializable_classes[class_name]
        return result

    def get_registered_class_name(self, serializable_cls):
        result = None
        if serializable_cls in self._serializable_class_names:
            result = self._serializable_class_names[serializable_cls]
        return result


PT_SERIALIZABLE_CLASSES = SerializableClassesRegistry()
TF_SERIALIZABLE_CLASSES = SerializableClassesRegistry()


class JSONSerializer:
    def __init__(self, serializable_classes: SerializableClassesRegistry):
        self._serializable_classes = serializable_classes
        self._encoder = self._Encoder(serializable_classes, indent=4, sort_keys=True)
        self._decoder = self._Decoder(serializable_classes)

    def serialize(self, obj) -> str:
        return self._encoder.encode(obj)

    def deserialize(self, json_string):
        return self._decoder.decode(json_string)

    class _Decoder(json.JSONDecoder):
        def __init__(self, serializable_classes: SerializableClassesRegistry, *args, **kwargs):
            super().__init__(object_hook=self._object_hook, *args, **kwargs)
            self._serializable_classes = serializable_classes

        def _object_hook(self, obj):
            if isinstance(obj, dict) and 'class_name' in obj:
                if obj['class_name'] == '__set__':
                    return set(obj['data'])
                elif obj['class_name'] == '__tuple__':
                    return tuple(obj['data'])
                elif 'dict' in obj:
                    class_name = obj['class_name']
                    json_dict = obj['dict']
                    serializable_cls = self._serializable_classes.get_registered_class(class_name)
                    if serializable_cls is None:
                        raise ValueError('Unknown class for deserialization: {}'.format(class_name))
                    if hasattr(serializable_cls, 'from_dict'):
                        return serializable_cls.from_dict(json_dict)
                    return serializable_cls(**json_dict)
                else:
                    raise ValueError('Unexpected format!')
            return obj

    class _Encoder(json.JSONEncoder):
        def __init__(self, serializable_classes: SerializableClassesRegistry, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._serializable_classes = serializable_classes

        def default(self, obj):
            """ serialize an arbitrary object to json compatible one """
            cls = obj.__class__
            registered_name = self._serializable_classes.get_registered_class_name(cls)
            if registered_name is not None:
                return {'class_name': registered_name, 'dict': obj.to_dict()}
            if isinstance(obj, set):
                return {'class_name': '__set__', 'data': list(obj)}
            if isinstance(obj, tuple):
                return {'class_name': '__tuple__', 'data': obj}

            # call default _encoder for other types of objects, will throw an error if it's not JSON-serializable
            return json.JSONEncoder.default(self, obj)

class CommonSerializableClasses:
    @staticmethod
    def register(name=None):
        return TF_SERIALIZABLE_CLASSES.register(name)(PT_SERIALIZABLE_CLASSES.register(name))


COMMON_SERIALIZABLE_CLASSES = CommonSerializableClasses
#
# class JSONHelper:
#     @staticmethod
#     @abstractmethod
#     def to_json_type(obj):
#         pass
#
#
#     def from_json_type(cls, json_type):
#         pass
# # TODO: register encode/decoder helpers for built_in types
#   BUILT_IN_SERIALIZERS.register(type='set')
#   if get_built_in_name(obj) in BUILT_IN_SERIALIZERS:
#       BUILT_IN_SERIALIZERS.get().encode()
#
# class SetSerializer(JSONHelper):
#     @staticmethod
#     def to_dict(obj):
#         if isinstance(obj, set):
#             return {'class_name': '__set__', 'data': list(obj)}
#
#     def from_dict(cls, obj):
#         if obj['class_name'] == '__set__':
#             return set(obj['data'])
#
# class TupleSerializer(JSONHelper):
#     return tuple(_decode_helper(i) for i in obj['items']
#
# def deserialize(obj_repr):
#     if (not isinstance(obj_repr, dict)) or ('class_name' not in obj_repr) or ('dict' not in obj_repr):
#         raise ValueError('Improper format: ' + str(obj_repr))
#
#     class_name = obj_repr['class_name']
#     cls = JSONSerializable.get_registered_object(class_name)
#     if cls is None:
#         raise ValueError('Unknown class for deserialization: {}'.format(class_name))
#     json_dict = obj_repr['dict']
#
#     deserialized_objects = {}
#     for key, item in json_dict.items():
#         deserialized_objects[key] = deserialize(item)
#
#     for key, item in deserialized_objects.items():
#         json_dict[key] = deserialized_objects[key]
#
#     return cls.from_config(json_dict)
#
#
# @classmethod
# def deserialize(cls, loaded_json: Any):
#     try:
#         # TODO: static from class??
#         cls._deserialize(loaded_json)
#     except Exception as ex:
#         raise RuntimeError('Failed to deserialize {} from the loaded json'.format(cls.__name__)) from ex
#
#
# class JSONSerializable(ABC):
#     # TODO: should be outside to not overcomplicate subclasses
#
#
#     # TODO: think about explicit split of TF & PT & common
#     #  register
#     def __init_subclass__(cls, reg_name=None, prefix='PT', **kwargs):
#         super().__init_subclass__(**kwargs)
#         class_name = reg_name if reg_name is not None else cls.__name__
#         registered_name = prefix + '>' + class_name
#
#         if registered_name in cls.REGISTERED_CLASSES:
#             raise ValueError(
#                 '%s has already been registered to %s' %
#                 (registered_name, cls.REGISTERED_CLASSES[registered_name]))
#
#         if cls in cls.REGISTERED_NAMES:
#             raise ValueError('%s has already been registered to %s' %
#                              (registered_name, cls.REGISTERED_NAMES[cls]))
#
#         cls._register(registered_name, cls)
#         cls.registered_name = registered_name
#
#     @classmethod
#     # @abstractmethod
#     def _register(cls, registered_name, registered_class):
#         cls.REGISTERED_NAMES[registered_class] = registered_name
#         cls.REGISTERED_CLASSES[registered_name] = registered_class
#
#     @classmethod
#     def get_registered_class(cls, class_name):
#         result = None
#         if class_name in cls.REGISTERED_CLASSES:
#             result = cls.REGISTERED_CLASSES[class_name]
#         return result
#
#     @abstractmethod
#     def to_dict(self) -> Dict:
#         """ Returns a JSON serializable Python dictionary that represents the object
#         :return Python dictionary"""
#         pass
#
#     @classmethod
#     def from_dict(cls, json_dict: Dict) -> 'JSONSerializable':
#         """ Instantiates the object from its JSON serializable representation - Python dictionary. This method is the
#         reverse of `to_dict`.
#         :param: json_dict: A Python dictionary - the output of to_dict
#         :return instance of the object """
#         return cls(**json_dict)
#
#
