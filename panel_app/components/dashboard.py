"""Main dashboard component"""

import panel as pn
import pandas as pd
import hvplot.pandas
import param


class Dashboard(param.Parameterized):
    """Main dashboard component with reactive updates"""
    
    data = param.DataFrame(doc="The data to display")
    selected_column = param.Selector(doc="Column to visualize")
    chart_type = param.Selector(
        default="bar",
        objects=["bar", "line", "scatter", "hist"],
        doc="Type of chart to display"
    )
    
    def __init__(self, data, **params):
        # Set up data and column choices
        self.data = data
        numeric_columns = data.select_dtypes(include=["float64", "int64"]).columns.tolist()
        
        if numeric_columns:
            self.param.selected_column.objects = numeric_columns
            self.selected_column = numeric_columns[0]
        
        super().__init__(**params)
        
    @param.depends("data", "selected_column", "chart_type")
    def create_plot(self):
        """Create the main plot based on current parameters"""
        if self.data is None or self.selected_column is None:
            return pn.pane.Markdown("## No data available")
        
        try:
            if self.chart_type == "bar":
                plot = self.data.hvplot.bar(
                    y=self.selected_column,
                    height=400,
                    responsive=True,
                    title=f"{self.selected_column} - Bar Chart"
                )
            elif self.chart_type == "line":
                plot = self.data.hvplot.line(
                    y=self.selected_column,
                    height=400,
                    responsive=True,
                    title=f"{self.selected_column} - Line Chart"
                )
            elif self.chart_type == "scatter":
                plot = self.data.hvplot.scatter(
                    y=self.selected_column,
                    height=400,
                    responsive=True,
                    title=f"{self.selected_column} - Scatter Plot"
                )
            else:  # hist
                plot = self.data[self.selected_column].hvplot.hist(
                    height=400,
                    responsive=True,
                    title=f"{self.selected_column} - Histogram"
                )
            return plot
        except Exception as e:
            return pn.pane.Alert(f"Error creating plot: {str(e)}", alert_type="danger")
    
    @param.depends("data")
    def create_table(self):
        """Create the data table"""
        if self.data is None:
            return pn.pane.Markdown("## No data available")
        
        return pn.widgets.Tabulator(
            self.data,
            pagination="remote",
            page_size=10,
            height=300,
            show_index=True,
            configuration={
                "columnDefaults": {
                    "tooltip": True,
                }
            }
        )
    
    @param.depends("data")
    def create_stats(self):
        """Create summary statistics"""
        if self.data is None:
            return pn.pane.Markdown("## No data available")
        
        stats_df = self.data.describe().round(2)
        return pn.pane.DataFrame(
            stats_df,
            width=400,
            index=True
        )
    
    def view(self):
        """Create the complete dashboard view"""
        return pn.Column(
            pn.Row(
                pn.Column(
                    "# Data Visualization",
                    self.create_plot,
                    sizing_mode="stretch_both",
                ),
                pn.Column(
                    "# Summary Statistics",
                    self.create_stats,
                    sizing_mode="stretch_both",
                ),
                sizing_mode="stretch_width",
            ),
            pn.layout.Divider(),
            pn.Column(
                "# Data Table",
                self.create_table,
                sizing_mode="stretch_width",
            ),
            sizing_mode="stretch_width",
        )
