class StudentTODO(NotImplementedError):
    """Only explicit exercise placeholders raise this exception."""


def todo(component, file, test):
    raise StudentTODO(
        f"{component} is not implemented yet.\n\nComplete:\n{file}\n\n"
        f"Then run:\npython -m pytest {test} --run-student -q"
    )
