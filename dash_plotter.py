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
                 options = [{'label': i, 'value': i} for i in tests],
                 value=' '),
            style = {'width': '48%', 'display': 'inline-block'}),
                 
    html.Div(dcc.Dropdown(id = 'ch_entry',
                 options = [{'label': i, 'value': i} for i in channels],
                 value=' '),
            style = {'width': '48%', 'display': 'inline-block', 'float': 'right'})]),
    html.Div([
    html.Div(dcc.Dropdown(id = 'tc_entry_2',
                 options = [{'label': i, 'value': i} for i in tests],
                 value=' '),
            style = {'width': '48%', 'display': 'inline-block'}),
                 
    html.Div(dcc.Dropdown(id = 'ch_entry_2',
                 options = [{'label': i, 'value': i} for i in channels],
                 value=' '),
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
    #initialize the new graph
    new_graph = {'data'  : [],
                 'layout': {'title'     : ' ',
                            'titlefont' : {'size': 20},
                            'showlegend': True}}
    
    #if no search in either fields, plot nothing
    #if [tc,ch]==[None,None] and [tc2,ch2]==[None,None]
    if (tc in [' ', None] or ch in [' ', None]) and (tc2 in [' ', None] or ch2 in [' ', None]):
#    if not(tc!=' ' and ch!=' ') and not(tc2!=' ' and ch2!=' '):
        new_graph['data'].append({'x': [], 'y': []})
        return new_graph
    
    # if first fields are not empty, then add to plot
    if not(tc in [' ', None]) and not(ch in [' ', None]):
        t, x = get_test(tc, ch)
        if (x[1:]==0).all():
            t = [0]
            x = [x[0]]
            new_graph['data'].append({'x': t, 'y': x, 'name': tc + ',' + ch, 'mode': 'markers','line': {'color': '#182952'}})
        else:
            new_graph['data'].append({'x': t, 'y': x, 'name': tc + ',' + ch,'line': {'color': '#182952'}})
    
    # if second fields are not empty, then add to plot
    if not(tc2 in [' ', None]) and not(ch2 in [' ', None]):
        t2, x2 = get_test(tc2,ch2)
        if (x2[1:]==0).all():
            t2 = [0]
            x2 = [x2[0]]
            new_graph['data'].append({'x': t2, 'y': x2, 'name': tc2 + ',' + ch2, 'mode': 'markers','line': {'color': '#ff5da2'}})
        else:
            new_graph['data'].append({'x': t2, 'y': x2, 'name': tc2 + ',' + ch2,'line': {'color': '#ff5da2'}})

    return new_graph

if __name__=='__main__':
    app.run_server()