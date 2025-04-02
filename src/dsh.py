
import sqlite3
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import find_peaks
from scipy.integrate import trapezoid
from astropy.modeling import models, fitting
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State
from dash import dash_table
from flask import Flask
import plotly.colors as pc
import numpy as np
import sqlite3
import pandas as pd
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import find_peaks


class DataFetcher:
    def __init__(self, db_nist, db_spectrum):
        self.db_nist = db_nist
        self.db_spectrum = db_spectrum

    def get_nist_data(self, element, sp_num):
        conn = sqlite3.connect(self.db_nist)
        query = """
            SELECT "obs_wl_air(nm)", "gA(s^-1)", "Ek(cm-1)", "Ei(cm-1)", "g_i", "g_k", "acc"
            FROM spectrum_data
            WHERE element = ? AND sp_num = ?
        """
        data = conn.execute(query, (element, sp_num)).fetchall()
        conn.close()
        return data

    def get_experimental_data(self, sample_name, iteration):
        conn = sqlite3.connect(self.db_spectrum)
        query = """
            SELECT wavelength, intensity
            FROM spectrum_data
            WHERE sample_name = ? AND iteration = ?
            ORDER BY wavelength
        """
        data = conn.execute(query, (sample_name, iteration)).fetchall()
        conn.close()
        if not data:
            return np.array([]), np.array([])
        wavelengths, intensities = zip(*data)
        return np.array(wavelengths, dtype=float), np.array(intensities, dtype=float)
    def get_peak_data(sample_name):
        excel_file = f"da/{sample_name}.xlsx"

        try:
          df = pd.read_excel(excel_file)
          return df
        except FileNotFoundError:
          print(f"Excel file not found: {excel_file}")
          return pd.DataFrame()
        except Exception as e:
          print(f"Error reading Excel file: {e}")
          return pd.DataFrame()

class SpectrumSimulator:
    def __init__(self, nist_data, temperature, resolution=24880):
        self.nist_data = nist_data
        self.temperature = temperature
        self.resolution = int(resolution)

    @staticmethod
    def partition_function(energy_levels, degeneracies, T):
        k_B = 8.617333262145e-5
        Z = np.sum([g * np.exp(-E / (k_B * T)) for g, E in zip(degeneracies, energy_levels)])
        return Z

    @staticmethod
    def calculate_intensity(T, energy, degeneracy, einstein_coeff, Z):
        k_B = 8.617333262145e-5
        return (degeneracy * np.exp(-energy / (k_B * T)) * einstein_coeff) / Z

    @staticmethod
    def gaussian_profile(x, center, sigma):
        return np.exp(-0.5 * ((x - center) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

    def simulate(self):
        wavelengths = np.linspace(200, 900, self.resolution)
        intensities = np.zeros_like(wavelengths)
        energy_levels = []
        degeneracies = []
        for wl, gA, Ek, Ei, gi, gk, acc in self.nist_data:
            if all(value is not None for value in [wl, gA, Ek, Ei, gi, gk]):
                try:
                    wl = float(wl)
                    gA = float(gA)
                    Ek, Ei = float(Ek), float(Ei)
                    Ek = Ek / 8065.544
                    Ei = Ei / 8065.544
                    gi = float(gi)
                    gk = float(gk)
                    energy_levels.extend([Ei, Ek])
                    degeneracies.extend([gi, gk])
                except ValueError:
                    continue
        Z = self.partition_function(energy_levels, degeneracies, self.temperature)
        for wl, gA, Ek, Ei, gi, gk, acc in self.nist_data:
            if all(value is not None for value in [wl, gA, Ek, Ei, gi, gk]):
                try:
                    wl = float(wl)
                    gA = float(gA)
                    Ek = float(Ek) / 8065.544
                    gi = float(gi)
                    gk = float(gk)
                    Aki = gA / gk
                    intensity = self.calculate_intensity(self.temperature, Ek, gk, Aki, Z)
                    sigma = 0.1
                    intensities += intensity * self.gaussian_profile(wavelengths, wl, sigma)
                except ValueError:
                    continue
        if intensities.size > 0:
            intensities = intensities / np.max(intensities)
        return wavelengths, intensities


def baseline_als(intensities, lam, p, niter):
    lam = lam *1e4
    L = len(intensities)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
    w = np.ones(L)
    for _ in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w * intensities)
        w = p * (intensities > z) + (1 - p) * (intensities < z)
    return z

server = Flask(__name__)
data_fetcher = DataFetcher(db_nist="data1.db", db_spectrum="tanah_vulkanik.db")

app1 = dash.Dash(__name__, server=server, routes_pathname_prefix='/app1/', external_stylesheets=[dbc.themes.CYBORG])
app2 = dash.Dash(__name__, server=server, routes_pathname_prefix='/app2/')
app3 = dash.Dash(__name__, server=server, routes_pathname_prefix='/app3/', external_stylesheets=[dbc.themes.CYBORG])

app1.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Spectrum Analyzer", className="text-center mb-4"),
            html.P("Aplikasi untuk menganalisis spektrum atom.", className="text-center"),
            html.A('App 2', href='/app2', className="btn btn-secondary mr-2"),
            html.A('App 3', href='/app3', className="btn btn-secondary mr-2"),
            dbc.Card([
                dbc.CardHeader("Input Parameters"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Enter Element:", className="form-label"),
                            dbc.Input(id='element-input', type='text', value='Ca', className="form-control")
                        ], className="mb-3"),
                        dbc.Col([
                            dbc.Label("Select Ion Stage:", className="form-label"),
                            dcc.Dropdown(
                                id='ion-stage-dropdown',
                                options=[{'label': str(i), 'value': i} for i in range(1, 6)],
                                value=1,
                                clearable=False,
                                className="form-control"
                            )
                        ], className="mb-3"),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Enter Temperature (K):", className="form-label"),
                            dbc.Input(id='temperature-input', type='number', value=9600, min=1000, step=100, className="form-control")
                        ], className="mb-3"),
                        dbc.Col([
                            dbc.Label("Max Intensity:", className="form-label"),
                            dbc.Input(id='max-intensity-input', type='number', value=1, step=0.000001, className="form-control")
                        ], className="mb-3"),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Select Sample:", className="form-label"),
                            dcc.Dropdown(
                                id='sample-dropdown',
                                options=[{'label': f"S{i}", 'value': f"S{i}"} for i in range(1, 25)],
                                value="S1",
                                clearable=False,
                                className="form-control"
                            )
                        ], className="mb-3"),
                        dbc.Col([
                            dbc.Label("Iteration:", className="form-label"),
                            dcc.Dropdown(
                                id='iteration-dropdown',
                                options=[{'label': f"{i}", 'value': f"{i}"} for i in range(1, 4)],
                                value="1",
                                clearable=False,
                                className="form-control"
                            )
                        ], className="mb-3"),
                    ]),
                    dbc.Row([
                        dbc.Col(
                            dbc.Button('Analyze', id='analyze-button', n_clicks=0, color="primary", className="mr-2"),
                        ),
                        dbc.Col(
                            dbc.Button('Calibrate', id='calibrate-button', n_clicks=0, color="primary"),
                        ),
                    ]),
                ]),
                dbc.Card([
                    dbc.CardHeader("Baseline Correction Parameters"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Lambda (1e4):", className="form-label"),
                                dbc.Input(id='lam-input', type='number', value=1, min=1, step=1, className="form-control")
                            ]),
                            dbc.Col([
                                dbc.Label("P (p):", className="form-label"),
                                dbc.Input(id='p-input', type='number', value=0.01, min=0, max=1, step=0.001, className="form-control")
                            ]),
                        ]),
                        dbc.Row(
                            dbc.Col([
                                dbc.Label("Number of Iterations (niter):", className="form-label"),
                                dbc.Input(id='niter-input', type='number', value=7, min=1, step=1, className="form-control")
                            ])
                        )
                    ])
                ]),
            ]),
            dbc.Card([
                dbc.CardHeader("Peak Finding Parameters"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("height :", className="form-label"),
                            dbc.Input(id='height-input', type='number', value=0, step=0.001, className="form-control")
                        ]),
                        dbc.Col([
                            dbc.Label("threshold :", className="form-label"),
                            dbc.Input(id='threshold-input', type='number', value=0.1, step=0.00001, className="form-control")
                        ]),
                    ]),
                    dbc.Row(
                        dbc.Col([
                            dbc.Label("prominence :", className="form-label"),
                            dbc.Input(id='prominence-input', type='number', value=0.00004, step=0.00001, className="form-control")
                        ])
                    )
                ])
            ]),
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Spectrum Plot"),
                dbc.CardBody(
                    dcc.Graph(id='spectrum-plot', style={'height': '600px'})
                )
            ]),
            dbc.Card([
                dbc.CardHeader("Peak Matching Table"),
                dbc.CardBody([
                    dbc.Button('Save to Excel', id='save-table-button', color="primary", className="mb-3"),
                    html.Div(
                        id='peak-table',
                        style={
                            'overflowY': 'auto',
                            'maxHeight': '900px',
                            'overflowX': 'hidden'
                        }
                    )
                ])
            ], className="mt-3"),
        ], width=8)
    ]),
    dbc.Container([
        dcc.ConfirmDialog(
            id='save-notification',
            message='Tabel peak matching berhasil disimpan!'
        ),
    ]),
    dcc.Store(id='peak-table-data'),
    dcc.Store(id='sample-name-store'),  # Menyimpan nama sampel di state
    dbc.Row([  # Tombol untuk menyimpan data spektrum
        dbc.Col([
            dbc.Button('Save Spectrum Data', id='save-spectrum-button', n_clicks=0, color="primary", className="mr-2"),
        ]),
    ]),
], fluid=True)
app2.layout = dbc.Container([
    html.H1("Spectrum Analyzer", className="text-center mb-4"),
    html.A('App 1', href='/app1', className="btn btn-secondary mr-2"),
    html.A('App 3', href='/app3', className="btn btn-secondary mr-2"),
    dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dbc.Label("Select Sample:", className="form-label"),
                    dcc.Dropdown(
                        id='sample-dropdown',
                        options=[{'label': f"S{i}", 'value': f"S{i}"} for i in range(1, 25)],
                        value="S1",
                        clearable=False,
                        className="form-control"
                    )
                ], className="mb-3"),
                dbc.Col([
                    dbc.Label("Iteration:", className="form-label"),
                    dcc.Dropdown(
                        id='iteration-dropdown',
                        options=[{'label': f"{i}", 'value': f"{i}"} for i in range(1, 4)],
                        value="1",
                        clearable=False,
                        className="form-control"
                    )
                ], className="mb-3"),
            ]),
            dbc.Row([
                dbc.Col(
                    dcc.Input(
                        id='intensity-threshold',
                        type='number',
                        placeholder='Enter Intensity Threshold',
                        min=0,
                        style={'marginRight': '10px', 'padding': '10px', 'width': '100%'}
                    ), width=6
                ),
                dbc.Col(
                    dcc.Input(
                        id='localized-range',
                        type='number',
                        placeholder='Enter Localized Range (nm)',
                        min=1,
                        step=1,
                        value=5,
                        style={'marginRight': '10px', 'padding': '10px', 'width': '100%'}
                    ), width=6
                ),
            ]),
            dbc.Row(
                dbc.Col(
                    html.Button(
                        'Plot Spectrum',
                        id='plot-button',
                        n_clicks=0,
                        style={'padding': '10px', 'backgroundColor': '#007BFF', 'color': 'white', 'border': 'none',
                               'width': '100%'}
                    )
                )
            )
        ])
    ], className="mb-4"),
    dbc.Spinner(html.Div(id="loading-output"), color="primary"),
    dbc.Card([
        dbc.CardBody([
            dcc.Graph(
                id='spectrum-plot',
                config={
                    'toImageButtonOptions': {
                        'format': 'png',
                        'filename': 'spectrum_plot',
                        'height': 600,
                        'width': 900,
                        'scale': 10
                    },
                    'displaylogo': False,
                    'modeBarButtonsToAdd': [
                        'drawline', 'drawopenpath', 'drawcircle', 'eraseshape'
                    ],
                    'modeBarButtonsToRemove': [
                        'zoom2d', 'pan2d', 'select2d', 'lasso2d'
                    ],
                    'scrollZoom': True
                },
                style={'height': '600px', 'width': '100%'}
            )
        ])
    ])
], fluid=True, style={'fontFamily': 'Arial, sans-serif', 'padding': '20px'})
app3.layout = html.Div([
    html.H1("3D Spectrum Simulation", style={'textAlign': 'center'}),
    html.Div([
        html.A('App 1', href='/app1', className="btn btn-secondary mr-2"),
        html.A('App 2', href='/app2', className="btn btn-secondary mr-2"),
    ], style={'textAlign': 'center', 'marginBottom': '20px'}),
    html.Div([
        html.Div([
            html.Label("Enter Element:", style={'fontWeight': 'bold'}),
            dcc.Input(id='element-input', type='text', value='Fe', style={'width': '100%'})
        ], style={'width': '300px', 'display': 'inline-block', 'marginRight': '20px'}),
        html.Div([
            html.Label("Select Ion Stage:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='ion-stage-dropdown',
                options=[{'label': str(i), 'value': i} for i in range(1, 6)],
                value=1,
                clearable=False,
                style={'width': '100%'}
            )
        ], style={'width': '200px', 'display': 'inline-block'}),
    ], style={'marginBottom': '20px', 'textAlign': 'center'}),
    html.Button('Plot 3D', id='plot-button', n_clicks=0, style={'display': 'block', 'margin': '0 auto'}),
    dcc.Graph(id='spectrum-plot-3d', style={'height': '600px', 'width': '100%', 'marginTop': '20px'})
], style={'maxWidth': '1200px', 'margin': '0 auto'})

@app1.callback(
    Output('spectrum-plot', 'figure'),
    Output('peak-table', 'children'),
    Output('peak-table-data', 'data'),
    Input('analyze-button', 'n_clicks'),
    State('element-input', 'value'),
    State('ion-stage-dropdown', 'value'),
    State('temperature-input', 'value'),
    State('max-intensity-input', 'value'),
    State('sample-dropdown', 'value'),
    State('iteration-dropdown', 'value'),
    State('lam-input', 'value'),
    State('p-input', 'value'),
    State('niter-input', 'value'),
    State('height-input', 'value'),
    State('threshold-input', 'value'),
    State('prominence-input', 'value'),
    prevent_initial_call=True
)
def update_and_analyze(n_clicks, element, sp_num, temperature, mx, sample_name, iteration, lam, p, niter, height, threshold, prominence):
    if n_clicks > 0:
        nist_data = data_fetcher.get_nist_data(element, sp_num)
        simulator = SpectrumSimulator(nist_data, temperature)
        wavelengths, intensities = simulator.simulate()
        intensities = intensities * mx
        wavelengths_exp, intensities_exp = data_fetcher.get_experimental_data(sample_name, iteration)
        if len(wavelengths_exp) == 0:
            return go.Figure().update_layout(title="No data found"), "No experimental data found for the selected sample and iteration."
        bs = baseline_als(intensities_exp, lam=lam, p=p, niter=niter)
        intensities_exp = intensities_exp - bs
        intensities_exp = intensities_exp / np.max(intensities_exp)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=wavelengths_exp, y=intensities_exp, mode='lines', name='Experimental Data'))
        fig.add_trace(go.Scatter(x=wavelengths, y=intensities, mode='lines', name='Simulated Spectrum'))
        fig.update_layout(
            title=f"Experimental Spectrum : {sample_name}",
            xaxis_title='Wavelength (nm)',
            yaxis_title='Normalized Intensity (a.u)',
            template='plotly_white'
        )
        peaks, _ = find_peaks(intensities_exp, height=height, prominence=prominence)
        exp_peaks = wavelengths_exp[peaks]
        matches = []
        nist_dict = {float(data[0]): data for data in nist_data}
        for exp_peak_wl in exp_peaks:
            try:
                closest_nist_wl = min(nist_dict, key=lambda x: abs(x - exp_peak_wl))
                if abs(closest_nist_wl - exp_peak_wl) <= threshold:
                    data = nist_dict[closest_nist_wl]
                    matches.append({
                        "Element": element,
                        "Ion Stage": sp_num,
                        "NIST WL (nm)": round(float(data[0]), 6),
                        "Exp WL (nm)": round(exp_peak_wl, 6),
                        "Aki (s^-1)": round(float(data[1]) / float(data[5]), 0) if data[1] != '' else None,
                        "Ei (eV)": float(data[3]) / 8065.544 if data[3] != '' else None,
                        "Ek (eV)": float(data[2]) / 8065.544 if data[2] != '' else None,
                        "gi": float(data[4]) if data[4] != '' else None,
                        "gk": float(data[5]) if data[5] != '' else None,
                        "Acc": str(data[6]) if data[6] is not None else None
                    })
            except (ValueError, KeyError) as e:
                print(f"Error saat memproses peak {exp_peak_wl}: {e}")
        num_matches = len(matches)
        table_children = [
            html.Div(f"Jumlah garis : {num_matches}", className="text-center mb-2"),
            dash_table.DataTable(
                columns=[
                    {"name": "Element", "id": "Element"},
                    {"name": "Ion Stage", "id": "Ion Stage"},
                    {"name": "NIST WL (nm)", "id": "NIST WL (nm)"},
                    {"name": "Exp WL (nm)", "id": "Exp WL (nm)"},
                    {"name": "Aki (s^-1)", "id": "Aki (s^-1)"},
                    {"name": "Ei (eV)", "id": "Ei (eV)"},
                    {"name": "Ek (eV)", "id": "Ek (eV)"},
                    {"name": "gi", "id": "gi"},
                    {"name": "gk", "id": "gk"},
                    {"name": "Acc", "id": "Acc"}
                ],
                data=matches,
                style_table={'overflowY': 'auto', 'maxHeight': '900px'},
                style_cell={'textAlign': 'center', 'padding': '5px'},
                style_header={'backgroundColor': 'black', 'color': 'white', 'fontWeight': 'bold'},
                page_size=100
            )
        ] if matches else html.P("Tidak ada peak yang cocok ditemukan.", style={'textAlign': 'center'})
        return fig, table_children, matches
    return go.Figure(), ""
@app1.callback(
    Output("save-notification", "displayed"),
    Input("save-spectrum-button", "n_clicks"),
    State("spectrum-plot", "figure"),  # Ambil data figure dari spectrum-plot
    State('sample-name-store', 'data'),  # Ambil nama sampel dari state
    prevent_initial_call=True,
)
def save_spectrum_data(n_clicks, figure, sample_name):
    if n_clicks is None:
        return False

    # Ambil panjang gelombang dari sumbu-x pada plot
    wavelengths = figure['data'][0]['x']

    # Buat dictionary untuk menyimpan data
    data = {'Wavelength': wavelengths}

    # Loop melalui setiap sampel
    for i in range(1, 25):  # Asumsi ada 24 sampel (S1 hingga S24)
        # Ambil data processed intensity untuk sampel ke-i
        processed_intensity = figure['data'][i]['y']  # Asumsikan processed intensity ada di trace ke-i

        # Tambahkan data ke dictionary
        data[f'S{i} - Processed Intensity'] = processed_intensity

    # Buat DataFrame dari dictionary
    df = pd.DataFrame(data)

    # Simpan data ke dalam file Excel
    df.to_excel("spectrum_data.xlsx", index=False)

    return True  # Tampilkan notifikasi setelah file tersimpan


@app1.callback(
    Output('save-table-button', 'n_clicks'),
    Output('save-notification', 'displayed'),
    Input('save-table-button', 'n_clicks'),
    State('peak-table-data', 'data'),
    State('sample-dropdown', 'value'),
    prevent_initial_call=True
)
def save_table(n_clicks, table_data, sample):
    if n_clicks is not None:
        df = pd.DataFrame(table_data)
        filename = f"da/{sample}.xlsx"
        try:
            existing_df = pd.read_excel(filename)
            df = pd.concat([existing_df, df], ignore_index=True)
        except FileNotFoundError:
            pass
        df.to_excel(filename, index=False)
        print(f"Tabel peak matching disimpan ke {filename}")
        return 0, True

@app2.callback(
    Output('spectrum-plot', 'figure'),
    Input('plot-button', 'n_clicks'),
    Input('sample-dropdown', 'value'),
    State('iteration-dropdown', 'value'),
    Input('intensity-threshold', 'value'),
    Input('localized-range', 'value')
)
def update_plot(n_clicks, sample, iteration, intensity_threshold, localized_range):
    if n_clicks > 0 and sample and localized_range:
        exp_wavelengths, exp_intensities = data_fetcher.get_experimental_data(sample, iteration)
        if exp_wavelengths.size == 0:
            return go.Figure()
        df = DataFetcher.get_peak_data(sample)
        if df.empty:
            return go.Figure()
        exp_peak_wl = df['Exp WL (nm)'].dropna().values
        elements = df['Element'].dropna().values
        ion_stages = df['Ion Stage'].dropna().values
        exp_intensity = []
        for peak_wl in exp_peak_wl:
            idx = np.argmin(np.abs(exp_wavelengths - peak_wl))
            exp_intensity.append(exp_intensities[idx])
        if intensity_threshold is not None:
            mask = np.array(exp_intensity) >= intensity_threshold
            filtered_peaks = zip(exp_peak_wl[mask], np.array(exp_intensity)[mask], elements[mask], ion_stages[mask])
        else:
            filtered_peaks = zip(exp_peak_wl, exp_intensity, elements, ion_stages)
        grouped_labels = {}
        for wl, intensity, element, ion_stage in filtered_peaks:
            key = (element, ion_stage, int(wl // localized_range * localized_range))
            if key not in grouped_labels:
                grouped_labels[key] = {}
            rounded_wl = round(wl, 2)
            if rounded_wl not in grouped_labels[key]:
                grouped_labels[key][rounded_wl] = intensity
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=exp_wavelengths, y=exp_intensities, mode='lines', name='Experimental Data', line=dict(color='black')))
        unique_elements = sorted(set(elements))
        colors = pc.qualitative.Set1
        element_colors = {element: colors[i % len(colors)] for i, element in enumerate(unique_elements)}
        for (element, ion_stage, wl_group), peaks in grouped_labels.items():
            wavelengths = list(peaks.keys())
            intensities = list(peaks.values())
            avg_wl = np.mean(wavelengths)
            avg_intensity = np.mean(intensities)
            ion_stage_roman = f"{['I', 'II', 'III', 'IV', 'V'][ion_stage-1]}" if ion_stage <= 5 else f"{ion_stage}"
            label_text = f"<b>{element} {ion_stage_roman}</b> " + ", ".join(f"{wl:.2f}" for wl in sorted(wavelengths)) + " nm"
            fig.add_annotation(
                x=avg_wl,
                y=avg_intensity + 0.1,
                text=label_text,
                showarrow=False,
                font=dict(size=12),
                bgcolor="rgba(0,0,0,0)",
                bordercolor="black",
                borderwidth=0,
                borderpad=4,
                opacity=1,
                textangle=-90,
                xanchor="center",
                yanchor="bottom"
            )
            for wl, intensity in peaks.items():
                fig.add_trace(go.Scatter(
                    x=[avg_wl, wl],
                    y=[avg_intensity + 0.1, intensity],
                    mode='lines',
                    line=dict(color=element_colors[element], width=0.7, dash='dash'),
                    showlegend=False,
                    hoverinfo='text',
                    text=f"{element}<br>Wavelength: {wl:.2f} nm<br>Intensity: {intensity:.2f}"
                ))
        fig.update_yaxes(range=[0, 1.2 * max(exp_intensities)])
        fig.update_layout(
            xaxis_title='Wavelength (nm)',
            yaxis_title='Normalized Intensity (a.u.)',
            plot_bgcolor='rgba(255,255,255,1)',
            paper_bgcolor='rgba(255,255,255,1)',
            showlegend=False,
            margin=dict(l=50, r=50, t=50, b=50),
            xaxis=dict(showline=True, linewidth=1, linecolor='black'),
            yaxis=dict(showline=True, linewidth=1, linecolor='black')
        )
        return fig
    return go.Figure()
@app3.callback(
    Output('spectrum-plot-3d', 'figure'),
    Input('plot-button', 'n_clicks'),
    State('element-input', 'value'),
    State('ion-stage-dropdown', 'value')
)
def plot_3d_spectrum(n_clicks, element, ion_stage):
    if n_clicks > 0:
        wavelengths = np.linspace(200, 900, 24880)
        temperatures = np.arange(5000, 1000, -200)
        X, Y = np.meshgrid(wavelengths, temperatures)

        nist_data = data_fetcher.get_nist_data(element, ion_stage)
        simulator = SpectrumSimulator(nist_data, temperature=10000)

        Z = np.zeros_like(X, dtype=float)
        for i, T in enumerate(temperatures):
            simulator.temperature = T
            _, intensities = simulator.simulate()
            Z[i, :] = intensities

        fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')])
        fig.update_layout(
            scene=dict(
                xaxis_title='Wavelength (nm)',
                yaxis_title='Temperature (K)',
                zaxis_title='Intensity'),
            title=f"Spektrum Simulasi {element} {ion_stage}",
            width=800,
            height=800
        )
        return fig
    return go.Figure()
@server.route('/', subdomain='<app_name>')
def route_to_app(app_name):
    if app_name == 'app1':
        return app1.index()
    elif app_name == 'app2':
        return app2.index()
    elif app_name == 'app3':
        return app3.index()
    else:
        return "Subdomain tidak valid"
if __name__ == '__main__':
    server.run(host='0.0.0.0', port=8056, debug=True, use_reloader=False)
