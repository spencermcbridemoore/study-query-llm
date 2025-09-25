"""Widget components for the sidebar"""

import panel as pn
import pandas as pd


def create_sidebar_widgets(dashboard):
    """Create and return sidebar widgets linked to the dashboard"""
    
    widgets = []
    
    # Column selector
    column_select = pn.widgets.Select(
        name='Select Column',
        value=dashboard.selected_column,
        options=dashboard.param.selected_column.objects,
        width=250,
    )
    column_select.link(dashboard, value='selected_column')
    widgets.append(column_select)
    
    # Chart type selector
    chart_select = pn.widgets.Select(
        name='Chart Type',
        value=dashboard.chart_type,
        options=dashboard.param.chart_type.objects,
        width=250,
    )
    chart_select.link(dashboard, value='chart_type')
    widgets.append(chart_select)
    
    # File upload widget
    file_input = pn.widgets.FileInput(
        accept='.csv',
        name='Upload CSV File',
        width=250,
    )
    
    def process_file(event):
        """Process uploaded file"""
        if file_input.value is not None:
            try:
                # Read the uploaded CSV
                df = pd.read_csv(file_input.file)
                dashboard.data = df
                
                # Update column selector
                numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
                if numeric_columns:
                    dashboard.param.selected_column.objects = numeric_columns
                    dashboard.selected_column = numeric_columns[0]
                    column_select.options = numeric_columns
                    column_select.value = numeric_columns[0]
                
                pn.state.notifications.success('File uploaded successfully!')
            except Exception as e:
                pn.state.notifications.error(f'Error uploading file: {str(e)}')
    
    file_input.param.watch(process_file, 'value')
    widgets.append(file_input)
    
    # Add a refresh button
    refresh_btn = pn.widgets.Button(
        name='Refresh Data',
        button_type='primary',
        width=250,
    )
    
    def refresh_data(event):
        """Refresh the dashboard data"""
        pn.state.notifications.info('Data refreshed!')
        # Trigger a refresh by updating the param
        dashboard.param.trigger('data')
    
    refresh_btn.on_click(refresh_data)
    widgets.append(refresh_btn)
    
    # Add a divider
    widgets.append(pn.layout.Divider())
    
    # Add theme toggle
    theme_toggle = pn.widgets.Toggle(
        name='Dark Theme',
        width=250,
    )
    
    def toggle_theme(event):
        """Toggle between light and dark themes"""
        if event.new:
            pn.config.theme = 'dark'
        else:
            pn.config.theme = 'default'
    
    theme_toggle.param.watch(toggle_theme, 'value')
    widgets.append(theme_toggle)
    
    return widgets