# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 16:57:13 2018
test dash
@author: tangk
Plot things interactively using Dash

"""

import dash
import dash_core_components as dcc
import dash_html_components as html
from PMG.read_data import get_test, get_test_info

tests, channels = get_test_info()
tests.sort()
channels.sort()
#%%


app = dash.Dash()

app.layout = html.Div(children=[
    dcc.Graph(
        id='time_series',
        figure={
            'data': [
                {'x': [], 'y': []}],
            'layout': {
                'title': ' ',
                'titlefont':{'size':20},
            'showlegend': True
            }
        },
        style={'height': '750'}
    ),
    
    html.Div([
    html.Div(dcc.Dropdown(id = 'tc_entry',
                 options = [{'label': i, 'value': i} for i in tests]),
            style = {'width': '48%', 'display': 'inline-block'}),
                 
    html.Div(dcc.Dropdown(id = 'ch_entry',
                 options = [{'label': i, 'value': i} for i in channels]),
            style = {'width': '48%', 'display': 'inline-block', 'float': 'right'})]),
    html.Div([
    html.Div(dcc.Dropdown(id = 'tc_entry_2',
                 options = [{'label': i, 'value': i} for i in tests]),
            style = {'width': '48%', 'display': 'inline-block'}),
                 
    html.Div(dcc.Dropdown(id = 'ch_entry_2',
                 options = [{'label': i, 'value': i} for i in channels]),
            style = {'width': '48%', 'display': 'inline-block', 'float': 'right'})])
#    dcc.Input(id='tc_entry', value=' ', type='text')
])



@app.callback(
        dash.dependencies.Output(component_id='time_series',component_property='figure'),
        [dash.dependencies.Input(component_id='tc_entry',component_property='value'),
         dash.dependencies.Input(component_id='ch_entry',component_property='value'),
         dash.dependencies.Input(component_id='tc_entry_2',component_property='value'),
         dash.dependencies.Input(component_id='ch_entry_2',component_property='value')])
def update_figure(tc,ch,tc2,ch2):
    if tc==None or ch==None or tc[0]==' ' or ch[0]==' ':
        new_graph = {
                'data': [{'x': [], 'y': []}],
                'layout': {
                        'title': ' ',
                        'titlefont':{'size':20}}}        
    elif tc2==None or ch2==None or tc2[0]== ' ' or ch2[0]==' ':
        # there is only one plot
        t, x = get_test(tc,ch)
        new_graph = {
                'data': [{'x':t,'y':x}],
                'layout':{
                        'title': tc + '\t' + ch,
                        'titlefont':{'size':20}},
                        'showlegend': True}
    else: 
        t, x = get_test(tc, ch)
        t2, x2 = get_test(tc2,ch2)
        new_graph = {'data': [{'x': t, 'y': x, 'name': tc + ',' + ch},
                              {'x': t2,'y': x2,'name': tc2 + ',' + ch2}],
                    'layout':{'title': ' ',
                              'showlegend': True}}
    return new_graph

if __name__=='__main__':
    app.run_server()