import sys
from types import ModuleType

def get_error_detail_massage(
    error_masg: object,
    error_detail_from_run_time: ModuleType,
) -> str:
    _, _, exc_tb = error_detail_from_run_time.exc_info()

    if exc_tb is None:
        return f"Error: {error_masg}"

    # Where the call was made.
    caller_file = exc_tb.tb_frame.f_code.co_filename
    caller_line = exc_tb.tb_lineno

    # Where the error actually happened.
    while exc_tb.tb_next is not None:
        exc_tb = exc_tb.tb_next

    error_file = exc_tb.tb_frame.f_code.co_filename
    error_line = exc_tb.tb_lineno

    return (
        f"Called from: [{caller_file}:{caller_line}]\n"
        f"Error occurred at: [{error_file}:{error_line}]\n"
        f"Message: {error_masg}"
    )
#=========================#
class CustomException(Exception):
    def __init__(
        self,
        error_massage_from_exception: object,
        error_detail_from_run_time: ModuleType,
    ):
        self.error_detail = get_error_detail_massage(
            error_masg=error_massage_from_exception,
            error_detail_from_run_time=error_detail_from_run_time,
        )
        super().__init__(self.error_detail)

    def __str__(self):
        return self.error_detail


if __name__ == "__main__":
    def test_function():
        return 10 / 0

    try:
        test_function()
    except Exception as e:
        raise CustomException(e, sys) from e