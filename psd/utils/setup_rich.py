from rich.traceback import install

install(
    show_locals=False,
    width=100,
    extra_lines=3,
    theme="lightbulb",
    word_wrap=True,
)
