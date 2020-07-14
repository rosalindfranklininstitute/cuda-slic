

import hug


###############################################################################
# Arg types

def wrap_output_type(obj, desc=None):
    obj.__desc__ = desc or obj.__class__.__name__
    return obj


String = wrap_output_type(hug.types.Text(), 'String')
DataURI = wrap_output_type(hug.types.Text(), 'DataURI')
DataURIList = wrap_output_type(hug.types.DelimitedList[str](','), 'DataURIList')

SmartBoolean = wrap_output_type(hug.types.SmartBoolean(), 'SmartBoolean')
Int = wrap_output_type(hug.types.number, 'Int')
Float = wrap_output_type(hug.types.float_number, 'Float')

IntList = wrap_output_type(hug.types.DelimitedList[int](','), 'IntList')
FloatList = wrap_output_type(hug.types.DelimitedList[float](','), 'FloatList')
IntOrVector = wrap_output_type(hug.types.multi(Int, IntList), 'IntOrVector')
FloatOrVector = wrap_output_type(hug.types.multi(Float, FloatList), 'FloatOrVector')
IntOrBool = wrap_output_type(hug.types.multi(SmartBoolean, Int, IntList), 'IntOrBool')


class IntOrNone(hug.types.Type):

    def __call__(self, value):
        if value is None or type(value) == int:
            return value
        return int(value)

IntOrNone = wrap_output_type(IntOrNone())


class IntListOrNone(hug.types.DelimitedList):

    def __call__(self, value):
        value = super().__call__(value)
        return list(map(int, value)) if value else None

IntListOrNone = wrap_output_type(IntListOrNone())
