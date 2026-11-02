from dataclasses import dataclass


@dataclass(frozen=True)
class TextTable:
    """
    Plain text table with left-aligned, width-fitted columns.

    Attributes
    ----------
    header : tuple[str, ...]
        The column titles.
    rows : tuple[tuple[str, ...], ...]
        The cells of each row.
    """

    header: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]

    def render(self) -> str:
        """
        Render the table.

        Returns
        -------
        str
            The table with one line per row, below the header and a separator.
        """
        all_rows: tuple[tuple[str, ...], ...] = (self.header, *self.rows)
        widths: list[int] = [
            max(len(row[position]) for row in all_rows)
            for position in range(len(self.header))
        ]
        separator: tuple[str, ...] = tuple("-" * width for width in widths)
        return "\n".join(
            "  ".join(
                cell.ljust(width) for cell, width in zip(row, widths, strict=True)
            ).rstrip()
            for row in (self.header, separator, *self.rows)
        )
