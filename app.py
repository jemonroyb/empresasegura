from pycaret.classification import load_model, predict_model
import numpy as np 
import pandas as pd 
import joblib
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import streamlit as st
from PIL import Image



def predict_quality(model, df):
    
    predictions_data = predict_model(estimator = model, data = df)
   
    return predictions_data['Label'][0]

# Load  model a 
model = joblib.load(open("Final_et.pkl","rb"))

def data_preprocessor(df):
    """this function preprocess the user input
        return type: pandas dataframe
    """
    #df.wine_type = df.wine_type.map({'white':0, 'red':1})
    return df

    

    

def visualize_confidence_level(prediction_proba):
    """
    this function uses matplotlib to create inference bar chart rendered with streamlit in real-time 
    return type : matplotlib bar chart  
    """
    data = (prediction_proba[0]*100).round(2)
    grad_percentage = pd.DataFrame(data = data,columns = ['Porcentage'],index = ['Est','Int','Int_Est','Rob','Rob_Est','Rob_Int','Rob_Int_Est'])
    ax = grad_percentage.plot(kind='barh', figsize=(7, 4), color='#0067e7', zorder=10, width=0.8)
    ax.legend().set_visible(False)
    ax.set_xlim(xmin=0, xmax=100)
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)

    ax.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")
    
    vals = ax.get_xticks()
    for tick in vals:
        ax.axvline(x=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)

    ax.set_xlabel(" Porcentage(%) Nivel de confianza", labelpad=2, weight='bold', size=12)
    ax.set_ylabel("Victimización", labelpad=10, weight='bold', size=12)
    ax.set_title('Nivel de confianza de la predicción ', fontdict=None, loc='center', pad=None, weight='bold')

    st.pyplot()
   
    return
    
add_selectbox = st.sidebar.selectbox(
    "Menú de navegación",
    ("MODELO","PREDICCIÓN"))


if add_selectbox == 'MODELO':   

    st.header("🦊 PREDICCIÓN DE DELICTIVOS EMPRESARIALES")
    st.write("Esta aplicación se puede usar para predecir los delitos empresariales basado en la encuesta nacional de victimización de empresas (ENVE-2018), elaborada por el Instituto Nacional de Estadística e Informática (INEI).")
    st.write("Los delitos más frecuentes son robo, fraude, estafa y extorsión. La inseguridad de las empresas afecta su productividad y consecuentemente a la competitividad de la misma.")

    st.sidebar.write("Selecciona la opción de navegación ☝️")                  
    st.sidebar.info('Jhon Monroy Barrios')
#read in wine image and render with streamlit
    st.markdown("<div align='center'><br>"
                "<img src='https://img.shields.io/badge/HECHO%20CON-PYTHON-red?style=for-the-badge'"
                "alt='API stability' height='25'/>"
                "<img src='https://img.shields.io/badge/LIBRER%C3%8DA-PYCARET-blue?style=for-the-badge'"
                "alt='API stability' height='25'/>"
                "<img src='https://img.shields.io/badge/INTERFAZ%20GR%C3%81FICA-STREAMLIT-brightgreen?style=for-the-badge'"
                "alt='API stability' height='25'/>"
                "<img src='https://img.shields.io/badge/SERVIDOR-HEROKU-yellow?style=for-the-badge'"
                "alt='API stability' height='25'/></div>"
                
                , unsafe_allow_html=True)
    image = Image.open('empresa.jpg')
    st.image(image, caption='',use_column_width=True)

   
    
    st.header("🎻 INSTRUCCIONES")
    st.subheader("♟ Navegación ♟")
    st.markdown("* Seleccione los valores a través de los controles del panel superior izquierdo para desplazarse por la aplicación y generar predicciones.")
    st.subheader("♟ Parámetros de entrada de usuario ♟")
    st.markdown("* Seleccione las variables de predicción a través de los controles del panel superior izquierdo para generar predicciones.")
    st.subheader("♟ Parámetros de Predicción ♟")
    st.markdown("* Resultados obtenidos automáticamente  de las variables de predicción. ")
    
    

if add_selectbox == 'PREDICCIÓN':

    def get_user_input():
        """
        this function is used to get user input using sidebar slider and selectbox 
        return type : pandas dataframe
        """
        st.sidebar.header('Parámetros de entrada') 
        acti2 = st.sidebar.selectbox('Código de Actividad Económica', ['ACABADO DE PRODUCTOS TEXTILES',
        'ACTIVIDADES COMBINADAS DE SERVICIOS ADMINISTRATIVOS DE OFICINA',                                                                                     
        'ACTIVIDADES CREATIVAS, ARTÍSTICAS Y DE ENTRETENIMIENTO',                                                                                                
        'ACTIVIDADES DE AGENCIAS DE COBRO Y AGENCIAS DE CALIFICACIÓN CREDITICIA',                                                                                 
        'ACTIVIDADES DE AGENCIAS DE EMPLEO',                                                                                                                      
        'ACTIVIDADES DE AGENCIAS DE VIAJES',                                                                                                                    
        'ACTIVIDADES DE AGENTES Y CORREDORES DE SEGUROS',                                                                                                        
        'ACTIVIDADES DE ALOJAMIENTO PARA ESTANCIAS CORTAS',                                                                                                    
        'ACTIVIDADES DE APOYO A LA ENSEÑANZA',                                                                                                                    
        'ACTIVIDADES DE APOYO PARA LA EXTRACCIÓN DE PETRÓLEO Y GAS NATURAL',                                                                                     
        'ACTIVIDADES DE APOYO PARA OTRAS ACTIVIDADES DE EXPLOTACIÓN DE MINAS Y CANTERAS',                                                                       
        'ACTIVIDADES DE ARQUITECTURA E INGENIERÍA Y ACTIVIDADES CONEXAS DE CONSULTORÍA TÉCNICA',                                                               
        'ACTIVIDADES DE ASOCIACIONES EMPRESARIALES Y DE EMPLEADORES',                                                                                            
        'ACTIVIDADES DE ASOCIACIONES PROFESIONALES',                                                                                                              
        'ACTIVIDADES DE ATENCIÓN DE ENFERMERÍA EN INSTITUCIONES',                                                                                                 
        'ACTIVIDADES DE BIBLIOTECAS Y ARCHIVOS',                                                                                                                  
        'ACTIVIDADES DE CENTROS DE LLAMADAS',                                                                                                                    
        'ACTIVIDADES DE CLUBES DEPORTIVOS',                                                                                                                      
        'ACTIVIDADES DE CONSULTORÍA DE GESTIÓN',                                                                                                                
        'ACTIVIDADES DE CONTABILIDAD, TENEDURÍA DE LIBROS Y AUDITORÍA; CONSULTORÍA FISCAL',                                                                      
        'ACTIVIDADES DE DESCONTAMINACIÓN Y OTROS SERVICIOS DE GESTIÓN DE DESECHOS',                                                                               
        'ACTIVIDADES DE DISTRIBUCIÓN DE PELÍCULAS CINEMATOGRÁFICAS, VÍDEOS Y PROGRAMAS DE TELEVISIÓN',                                                            
        'ACTIVIDADES DE ENVASADO Y EMPAQUETADO',                                                                                                                  
        'ACTIVIDADES DE EXHIBICIÓN DE PELÍCULAS CINEMATOGRÁFICAS Y CINTAS DE VÍDEO',                                                                              
        'ACTIVIDADES DE FOTOGRAFÍA',                                                                                                                              
        'ACTIVIDADES DE GESTIÓN DE FONDOS',                                                                                                                      
        'ACTIVIDADES DE HOSPITALES',                                                                                                                             
        'ACTIVIDADES DE INVESTIGACIÓN',                                                                                                                           
        'ACTIVIDADES DE JARDINES BOTÁNICOS Y ZOOLÓGICOS Y RESERVAS NATURALES',                                                                                    
        'ACTIVIDADES DE JUEGOS DE AZAR Y APUESTAS',                                                                                                              
        'ACTIVIDADES DE MENSAJERÍA',                                                                                                                             
        'ACTIVIDADES DE MUSEOS Y GESTIÓN DE LUGARES Y EDIFICIOS HISTÓRICOS',                                                                                      
        'ACTIVIDADES DE MÉDICOS Y ODONTÓLOGOS',                                                                                                                  
        'ACTIVIDADES DE OFICINAS CENTRALES',                                                                                                                      
        'ACTIVIDADES DE OPERADORES TURÍSTICOS',                                                                                                                  
        'ACTIVIDADES DE ORGANIZACIONES RELIGIOSAS',                                                                                                               
        'ACTIVIDADES DE OTRAS ASOCIACIONES N.C.P.',                                                                                                              
        'ACTIVIDADES DE PARQUES DE ATRACCIONES Y PARQUES TEMÁTICOS',                                                                                             
        'ACTIVIDADES DE PRODUCCIÓN DE PELÍCULAS CINEMATOGRÁFICAS, VÍDEOS Y PROGRAMAS DE TELEVISIÓN',                                                             
        'ACTIVIDADES DE RESTAURANTES Y DE SERVICIO MÓVIL DE COMIDAS',                                                                                           
        'ACTIVIDADES DE SEGURIDAD PRIVADA',                                                                                                                      
        'ACTIVIDADES DE SERVICIO DE BEBIDAS',                                                                                                                     
        'ACTIVIDADES DE SERVICIO DE SISTEMAS DE SEGURIDAD',                                                                                                       
        'ACTIVIDADES DE SERVICIOS RELACIONADAS CON LA IMPRESIÓN',                                                                                               
        'ACTIVIDADES DE SERVICIOS VINCULADAS AL TRANSPORTE ACUÁTICO',                                                                                             
        'ACTIVIDADES DE SERVICIOS VINCULADAS AL TRANSPORTE AÉREO',                                                                                                
        'ACTIVIDADES DE SERVICIOS VINCULADAS AL TRANSPORTE TERRESTRE',                                                                                            
        'ACTIVIDADES DE TELECOMUNICACIONES ALÁMBRICAS',                                                                                                          
        'ACTIVIDADES DE TELECOMUNICACIONES INALÁMBRICAS',                                                                                                         
        'ACTIVIDADES DE TELECOMUNICACIONES POR SATÉLITE.',                                                                                                        
        'ACTIVIDADES ESPECIALIZADAS DE DISEÑO',                                                                                                                   
        'ACTIVIDADES INMOBILIARIAS REALIZADAS A CAMBIO DE UNA RETRIBUCIÓN O POR CONTRATA',                                                                       
        'ACTIVIDADES INMOBILIARIAS REALIZADAS CON BIENES PROPIOS O ARRENDADOS',                                                                                 
        'ACTIVIDADES JURÍDICAS',                                                                                                                                 
        'ACTIVIDADES POSTALES',                                                                                                                                   
        'ACTIVIDADES VETERINARIAS',                                                                                                                               
        'ACUICULTURA DE AGUA DULCE',                                                                                                                              
        'ACUICULTURA MARÍTIMA',                                                                                                                                  
        'ADMINISTRACIÓN DE MERCADOS FINANCIEROS',                                                                                                                 
        'ALMACENAMIENTO Y DEPÓSITO',                                                                                                                             
        'ALQUILER Y ARRENDAMIENTO DE OTROS EFECTOS PERSONALES Y ENSERES DOMÉSTICOS',                                                                              
        'ALQUILER Y ARRENDAMIENTO DE OTROS TIPOS DE MAQUINARIA, EQUIPO Y BIENES TANGIBLES',                                                                     
        'ALQUILER Y ARRENDAMIENTO DE VEHÍCULOS AUTOMOTORES',                                                                                                     
        'ARRENDAMIENTO DE PROPIEDAD INTELECTUAL Y PRODUCTOS SIMILARES, EXCEPTO OBRAS PROTEGIDAS POR DERECHOS DE AUTOR',                                           
        'ARRENDAMIENTO FINANCIERO',                                                                                                                               
        'ASERRADOS Y ACEPILLADURA DE MADERA',                                                                                                                    
        'CAPTACIÓN, TRATAMIENTO Y DISTRIBUCIÓN DE AGUA',                                                                                                         
        'CONSTRUCCIÓN DE BUQUES Y ESTRUCTURAS FLOTANTES',                                                                                                         
        'CONSTRUCCIÓN DE CARRETERAS Y LÍNEAS DE FERROCARRIL',                                                                                                    
        'CONSTRUCCIÓN DE EDIFICIOS',                                                                                                                           
        'CONSTRUCCIÓN DE OTRAS OBRAS DE INGENIERÍA CIVIL',                                                                                                       
        'CONSTRUCCIÓN DE PROYECTOS DE SERVICIO PÚBLICO',                                                                                                         
        'CONSULTORÍA DE INFORMÁTICA Y DE GESTIÓN DE INSTALACIONES INFORMÁTICAS',                                                                                 
        'CORRETAJE DE VALORES Y DE CONTRATOS DE PRODUCTOS BÁSICOS',                                                                                              
        'CORTE, TALLA Y ACABADO DE LA PIEDRA',                                                                                                                    
        'CURTIDO Y ADOBO DE CUEROS',                                                                                                                              
        'DESTILACIÓN, RECTIFICACIÓN Y MEZCLA DE BEBIDAS ALCOHÓLICAS',                                                                                             
        'EDICIÓN DE LIBROS',                                                                                                                                     
        'EDICIÓN DE PERIÓDICOS, REVISTAS Y OTRAS PUBLICACIONES PERIÓDICAS',                                                                                      
        'EDUCACIÓN DEPORTIVA Y RECREATIVA',                                                                                                                       
        'ELABORACIÒN Y CONSERVACIÓN DE CARNE',                                                                                                                   
        'ELABORACIÒN Y CONSERVACIÓN DE FRUTAS,LEGUMBRES Y HORTALIZAS',                                                                                           
        'ELABORACIÒN Y CONSERVACIÓN DE PESCADOS, CRUSTÁCEOS Y MOLUSCOS',                                                                                         
        'ELABORACIÓN DE ACEITES Y GRASAS DE ORIGEN VEGETAL Y ANIMAL',                                                                                            
        'ELABORACIÓN DE AZÚCAR',                                                                                                                                  
        'ELABORACIÓN DE BEBIDAS MALTEADAS Y DE MALTA',                                                                                                            
        'ELABORACIÓN DE BEBIDAS NO ALCOHÓLICAS',                                                                                                                 
        'ELABORACIÓN DE CACAO Y CHOCOLATE Y DE PRODUCTOS DE CONFITERÍA',                                                                                         
        'ELABORACIÓN DE COMIDAS Y PLATOS PREPARADOS',                                                                                                            
        'ELABORACIÓN DE MACARRONES, FIDEOS, ALCUZCUS Y PRODUCTOS FARINÁCEOS SIMILARES',                                                                           
        'ELABORACIÓN DE OTROS PRODUCTOS ALIMENTICIOS N.C.P.',                                                                                                    
        'ELABORACIÓN DE PIENSOS PREPARADOS PARA ANIMALES',                                                                                                       
        'ELABORACIÓN DE PRODUCTOS DE MOLINERÍA.',                                                                                                                
        'ELABORACIÓN DE PRODUCTOS DE PANADERÍA',                                                                                                                
        'ELABORACIÓN DE PRODUCTOS LÁCTEOS',                                                                                                                      
        'ELABORACIÓN DE VINOS',                                                                                                                                   
        'ENSAYOS Y ANÁLISIS TÉCNICOS',                                                                                                                           
        'ENSEÑANZA CULTURAL',                                                                                                                                     
        'ENSEÑANZA PREESCOLAR Y PRIMARIA',                                                                                                                      
        'ENSEÑANZA SECUNDARIA DE FORMACIÓN GENERAL',                                                                                                            
        'ENSEÑANZA SECUNDARIA DE FORMACIÓN TÉCNICA Y PROFESIONAL',                                                                                                
        'ENSEÑANZA SUPERIOR',                                                                                                                                    
        'ESTUDIOS DE MERCADO Y ENCUESTAS DE OPINIÓN PÚBLICA',                                                                                                     
        'EVACUACIÓN DE AGUAS RESIDUALES',                                                                                                                         
        'EXPLOTACIÓN DE OTRAS MINAS Y CANTERAS N.C.P.',                                                                                                         
        'EXTRACCIÓN DE CARBÓN DE PIEDRA',                                                                                                                         
        'EXTRACCIÓN DE GAS NATURAL',                                                                                                                              
        'EXTRACCIÓN DE MINERALES DE HIERRO',                                                                                                                      
        'EXTRACCIÓN DE MINERALES PARA LA FABRICACIÓN DE ABONOS Y PRODUCTOS QUÍMICOS',                                                                             
        'EXTRACCIÓN DE OTROS MINERALES METALÍFEROS NO FERROSOS',                                                                                                
        'EXTRACCIÓN DE PETRÓLEO CRUDO',                                                                                                                           
        'EXTRACCIÓN DE PIEDRA, ARENA Y ARCILLA',                                                                                                                 
        'EXTRACCIÓN DE SAL',                                                                                                                                      
        'FABRICACIÓN ABONOS Y COMPUESTOS DE NITRÓGENO',                                                                                                           
        'FABRICACIÓN DE APARATOS DE USO DOMÉSTICO',                                                                                                               
        'FABRICACIÓN DE ARTICULOS DE PUNTO Y GANCHILLO',                                                                                                          
        'FABRICACIÓN DE ARTÍCULOS CONFECCIONADOS DE MATERIALES TEXTILES, EXCEPTO PRENDAS DE VESTIR',                                                              
        'FABRICACIÓN DE ARTÍCULOS DE CUCHILLERÍA, HERRAMIENTAS DE MANO Y ARTÍCULOS DE FERRETERÍA',                                                                
        'FABRICACIÓN DE ARTÍCULOS DE DEPORTE',                                                                                                                    
        'FABRICACIÓN DE ARTÍCULOS DE HORMIGÓN, DE CEMENTO Y DE YESO',                                                                                            
        'FABRICACIÓN DE ARTÍCULOS DE PIEL',                                                                                                                       
        'FABRICACIÓN DE BICICLETAS Y DE SILLONES DE RUEDAS PARA INVÁLIDOS',                                                                                       
        'FABRICACIÓN DE BISUTERÍA Y ARTÍCULOS CONEXOS',                                                                                                           
        'FABRICACIÓN DE BOMBAS, COMPRESORES, GRIFOS Y VÁLVULAS',                                                                                                  
        'FABRICACIÓN DE CALZADO',                                                                                                                                
        'FABRICACIÓN DE CARROCERÍAS PARA VEHÍCULOS AUTOMOTORES',                                                                                                 
        'FABRICACIÓN DE CEMENTO, CAL Y YESO',                                                                                                                    
        'FABRICACIÓN DE COMPONENTES Y TABLEROS ELECTRÓNICOS',                                                                                                     
        'FABRICACIÓN DE CUBIERTAS Y CÁMARAS DE CAUCHO',                                                                                                           
        'FABRICACIÓN DE CUERDAS, CORDELES, BRAMANTES Y REDES',                                                                                                    
        'FABRICACIÓN DE EQUIPO DE ELEVACIÓN Y MANIPULACIÓN',                                                                                                      
        'FABRICACIÓN DE EQUIPO DE IRRADIACIÓN Y EQUIPO ELECTRÓNICO DE USO MÉDICO Y TERAPÉUTICO',                                                                  
        'FABRICACIÓN DE EQUIPO ELÉCTRICO DE ILUMINACIÓN',                                                                                                         
        'FABRICACIÓN DE FIBRAS ARTIFICIALES',                                                                                                                     
        'FABRICACIÓN DE HERRAMIENTAS DE MANO MOTORIZADAS',                                                                                                        
        'FABRICACIÓN DE HOJAS DE MADERA PARA ENCHAPADO Y TABLEROS A BASE DE MADERA',                                                                             
        'FABRICACIÓN DE INSTRUMENTOS Y MATERIALES MÉDICOS Y ODONTOLÓGICOS',                                                                                       
        'FABRICACIÓN DE INSTRUMENTOS ÓPTICOS Y EQUIPO FOTOGRÁFICO',                                                                                               
        'FABRICACIÓN DE JABONES Y DETERGENTES, PREPARADOS PARA LIMPIAR Y PULIR, PERFUMES Y PREPARADOS DE TOCADOR.',                                              
        'FABRICACIÓN DE JOYAS Y ARTÍCULOS CONEXOS',                                                                                                               
        'FABRICACIÓN DE JUEGOS Y JUGUETES',                                                                                                                       
        'FABRICACIÓN DE MALETAS, BOLSOS DE MANO, Y ARTÍCULOS SIMILARES,Y DE ARTICULOS DE TALABARTERÍA Y GUARNICIONERÍA',                                          
        'FABRICACIÓN DE MAQUINARIA AGROPECUARIA Y FORESTAL',                                                                                                      
        'FABRICACIÓN DE MAQUINARIA METALÚRGICA',                                                                                                                  
        'FABRICACIÓN DE MAQUINARIA PARA EXPLOTACIÓN DE MINAS Y CANTERAS Y PARA OBRAS DE CONSTRUCCIÓN',                                                            
        'FABRICACIÓN DE MAQUINARIA PARA LA ELABORACIÓN DE ALIMENTOS, BEBIDAS Y TABACO',                                                                           
        'FABRICACIÓN DE MATERIALES DE CONSTRUCCIÓN DE ARCILLA',                                                                                                  
        'FABRICACIÓN DE MOTOCICLETAS',                                                                                                                            
        'FABRICACIÓN DE MOTORES Y TURBINAS, EXCEPTO MOTORES PARA AERONAVES, VEHÍCULOS AUTOMOTORES Y MOTOCICLETAS',                                                
        'FABRICACIÓN DE MOTORES, GENERADORES Y TRANSFORMADORES ELÉCTRICOS Y APARATOS DE DISTRIBUCIÓN Y CONTROL DE LA ENERGÍA ELÉCTRICA',                         
        'FABRICACIÓN DE MUEBLES',                                                                                                                                
        'FABRICACIÓN DE OTROS ARTÍCULOS DEL PAPEL Y CARTÓN',                                                                                                     
        'FABRICACIÓN DE OTROS HILOS Y CABLES ELÉCTRICOS',                                                                                                         
        'FABRICACIÓN DE OTROS PRODUCTOS DE CAUCHO',                                                                                                              
        'FABRICACIÓN DE OTROS PRODUCTOS DE MADERA; FABRICACIÓN DE ARTÍCULOS DE CORCHO, PAJA Y MATERIALES TRENZABLES.',                                           
        'FABRICACIÓN DE OTROS PRODUCTOS DE PORCELANA Y DE CERÁMICA',                                                                                              
        'FABRICACIÓN DE OTROS PRODUCTOS ELABORADOS DE METAL N.C.P.',                                                                                             
        'FABRICACIÓN DE OTROS PRODUCTOS MINERALES NO METÁLICOS N.C.P.',                                                                                          
        'FABRICACIÓN DE OTROS PRODUCTOS QUÍMICOS N.C.P.',                                                                                                        
        'FABRICACIÓN DE OTROS PRODUCTOS TEXTILES N.C.P.',                                                                                                        
        'FABRICACIÓN DE OTROS TIPOS DE EQUIPO DE TRANSPORTE N.C.P.',                                                                                              
        'FABRICACIÓN DE OTROS TIPOS DE EQUIPO ELÉCTRICO',                                                                                                         
        'FABRICACIÓN DE OTROS TIPOS DE MAQUINARIA DE USO ESPECIAL',                                                                                               
        'FABRICACIÓN DE OTROS TIPOS DE MAQUINARIA DE USO GENERAL',                                                                                               
        'FABRICACIÓN DE PARTES Y PIEZAS DE CARPINTERÍA PARA EDIFICIOS Y CONSTRUCCIONES',                                                                          
        'FABRICACIÓN DE PARTES, PIEZAS Y ACCESORIOS PARA VEHÍCULOS DE AUTOMOTORES',                                                                              
        'FABRICACIÓN DE PASTA DE MADERA, PAPEL Y CARTÓN',                                                                                                         
        'FABRICACIÓN DE PILAS, BATERÍAS Y ACUMULADORES',                                                                                                          
        'FABRICACIÓN DE PINTURAS, BARNICES Y PRODUCTOS DE REVESTIMIENTO SIMILARES, TINTAS DE IMPRENTA Y MASILLAS',                                               
        'FABRICACIÓN DE PLAGUICIDAS Y OTROS PRODUCTOS QUÍMICOS DE USO AGROPECUARIO',                                                                              
        'FABRICACIÓN DE PLÁSTICOS Y DE CAUCHO SINTÉTICO EN FORMAS PRIMARIAS',                                                                                     
        'FABRICACIÓN DE PRENDAS DE VESTIR, EXCEPTO PRENDAS DE PIEL',                                                                                            
        'FABRICACIÓN DE PRODUCTOS DE LA REFINACIÓN DEL PETRÓLEO',                                                                                                 
        'FABRICACIÓN DE PRODUCTOS DE PLÁSTICO',                                                                                                                 
        'FABRICACIÓN DE PRODUCTOS FARMACÉUTICOS, SUSTANCIAS QUÍMICAS MEDICINALES Y PRODUCTOS BOTÁNICOS DE USO FARMACÉUTICO',                                     
        'FABRICACIÓN DE PRODUCTOS METÁLICOS PARA USO ESTRUCTURAL',                                                                                              
        'FABRICACIÓN DE PRODUCTOS PRIMARIOS DE METALES PRECIOSOS Y OTROS METALES NO FERROSOS',                                                                    
        'FABRICACIÓN DE PRODUCTOS REFRACTARIOS',                                                                                                                  
        'FABRICACIÓN DE RECIPIENTES DE MADERA',                                                                                                                   
        'FABRICACIÓN DE SUSTANCIAS QUÍMICAS BÁSICAS',                                                                                                            
        'FABRICACIÓN DE TANQUES, DEPÓSITOS Y RECIPIENTES DE METAL',                                                                                               
        'FABRICACIÓN DE TAPICES Y ALFOMBRAS',                                                                                                                     
        'FABRICACIÓN DE TEJIDOS DE PUNTO Y GANCHILLO',                                                                                                            
        'FABRICACIÓN DE VEHÍCULOS AUTOMOTORES',                                                                                                                   
        'FABRICACIÓN DE VIDRIO Y DE PRODUCTOS DE VIDRIO',                                                                                                        
        'FABRICACIÓN DEL GAS',                                                                                                                                   
        'FABRICACIÓN DEL PAPEL Y CARTÓN ONDULADO Y DE ENVASES DE PAPEL Y CARTÓN',                                                                                 
        'FONDOS DE PENSIONES',                                                                                                                                    
        'FONDOS Y SOCIEDADES DE INVERSIÓN Y ENTIDADES FINANCIERAS SIMILARES',                                                                                     
        'FORJA, PRENSADO, ESTAMPADO Y LAMINADO DE METALES; PULVIMETALURGIA',                                                                                     
        'FOTOCOPIADO, PREPARACIÓN DE DOCUMENTOS Y OTRAS ACTIVIDADES ESPECIALIZADAS DE APOYO DE OFICINA',                                                          
        'FUNDICIÓN DE HIERRO Y ACERO',                                                                                                                            
        'FUNDICIÓN DE METALES NO FERROSOS',                                                                                                                       
        'GENERACIÓN, TRANSMISIÓN Y DISTRIBUCIÓN DE ENERGÍA ELÉCTRICA',                                                                                          
        'GESTIÓN DE INSTALACIONES DEPORTIVAS',                                                                                                                    
        'IMPRESIÓN',                                                                                                                                             
        'INDUSTRIAS BÁSICAS DE HIERRO Y ACERO',                                                                                                                   
        'INSTALACIONES DE FONTANERÍA, CALEFACCIÓN Y AIRE ACONDICIONADO',                                                                                         
        'INSTALACIONES ELÉCTRICAS',                                                                                                                              
        'INSTALACIÓN DE MAQUINARIA Y EQUIPO INDUSTRIALES',                                                                                                        
        'INVESTIGACIÓN Y DESARROLLO EXPERIMENTAL EN EL CAMPO DE LAS CIENCIAS NATURALES Y LA INGENIERÍA',                                                          
        'INVESTIGACIÓN Y DESARROLLO EXPERIMENTAL EN EL CAMPO DE LAS CIENCIAS SOCIALES Y LAS HUMANIDADES',                                                         
        'LAVADO Y LIMPIEZA, INCLUIDA LA LIMPIEZA EN SECO, DE PRODUCTOS TEXTILES Y DE PIEL',                                                                      
        'LIMPIEZA GENERAL DE EDIFICIOS',                                                                                                                         
        'MANIPULACIÓN DE CARGA',                                                                                                                                 
        'MANTENIMIENTO Y REPARACIÓN DE VEHÍCULOS AUTOMOTORES',                                                                                                   
        'ORGANIZACIÓN DE CONVENCIONES Y EXPOSICIONES COMERCIALES',                                                                                                
        'OTRAS ACTIVIDADES AUXILIARES DE LAS ACTIVIDADES DE SEGUROS Y FONDOS DE PENSIONES',                                                                      
        'OTRAS ACTIVIDADES AUXILIARES DE LAS ACTIVIDADES DE SERVICIOS FINANCIEROS',                                                                              
        'OTRAS ACTIVIDADES DE ALOJAMIENTO',                                                                                                                      
        'OTRAS ACTIVIDADES DE APOYO AL TRANSPORTE',                                                                                                             
        'OTRAS ACTIVIDADES DE ASISTENCIA SOCIAL SIN ALOJAMIENTO',                                                                                                 
        'OTRAS ACTIVIDADES DE ATENCIÓN DE LA SALUD HUMANA',                                                                                                     
        'OTRAS ACTIVIDADES DE ATENCIÓN EN INSTITUCIONES',                                                                                                         
        'OTRAS ACTIVIDADES DE CONCESIÓN DE CRÉDITO',                                                                                                              
        'OTRAS ACTIVIDADES DE DOTACIÓN DE RECURSOS HUMANOS',                                                                                                     
        'OTRAS ACTIVIDADES DE EDICIÓN',                                                                                                                           
        'OTRAS ACTIVIDADES DE ESPARCIMIENTO Y RECREATIVAS N.C.P.',                                                                                               
        'OTRAS ACTIVIDADES DE LIMPIEZA DE EDIFICIOS E INSTALACIONES INDUSTRIALES',                                                                                
        'OTRAS ACTIVIDADES DE SERVICIO DE COMIDAS',                                                                                                              
        'OTRAS ACTIVIDADES DE SERVICIOS DE APOYO A LAS EMPRESAS N.C.P',                                                                                         
        'OTRAS ACTIVIDADES DE SERVICIOS DE INFORMACIÓN N.C.P.',                                                                                                   
        'OTRAS ACTIVIDADES DE SERVICIOS FINANCIEROS, EXCEPTO LAS DE SEGUROS Y FONDOS DE PENSIONES, N.C.P.',                                                    
        'OTRAS ACTIVIDADES DE SERVICIOS PERSONALES N.C.P.',                                                                                                     
        'OTRAS ACTIVIDADES DE TECNOLOGÍA DE LA INFORMACIÓN Y DE SERVICIOS INFORMÁTICOS',                                                                        
        'OTRAS ACTIVIDADES DE TELECOMUNICACIÓN.',                                                                                                               
        'OTRAS ACTIVIDADES DE TRANSPORTE POR VÍA TERRESTRE',                            
        'OTRAS ACTIVIDADES DE VENTA AL POR MENOR EN COMERCIOS NO ESPECIALIZADOS',                                                                                
        'OTRAS ACTIVIDADES DE VENTA AL POR MENOR NO REALIZADAS EN COMERCIOS, PUESTOS DE VENTA O MERCADOS',                                                       
        'OTRAS ACTIVIDADES DEPORTIVAS',                                                                                                                          
        'OTRAS ACTIVIDADES ESPECIALIZADAS DE LA CONSTRUCCIÓN',                                                                                                   
        'OTRAS ACTIVIDADES PROFESIONALES, CIENTÍFICAS Y TÉCNICAS N.C.P.',                                                                                        
        'OTRAS INDUSTRIAS MANUFACTURERAS N.C.P.',                                                                                                                
        'OTRAS INSTALACIONES PARA OBRAS DE CONSTRUCCIÓN',                                                                                                        
        'OTROS SERVICIOS DE RESERVAS Y ACTIVIDADES CONEXAS',                                                                                                      
        'OTROS TIPOS DE ENSEÑANZA N.C.P.',                                                                                                                       
        'OTROS TIPOS DE INTERMEDIACIÓN MONETARIA.',                                                                                                              
        'PELUQUERÍA Y OTROS TRATAMIENTOS DE BELLEZA',                                                                                                             
        'PESCA DE AGUA DULCE',                                                                                                                                    
        'PESCA MARÍTIMA',                                                                                                                                       
        'POMPAS FÚNEBRES Y ACTIVIDADES CONEXAS',                                                                                                                 
        'PORTALES WEB',                                                                                                                                           
        'PREPARACIÓN DEL TERRENO',                                                                                                                               
        'PREPARACIÓN E HILATURA DE FIBRAS TEXTILES',                                                                                                             
        'PROCESAMIENTO DE DATOS, HOSPEDAJE Y ACTIVIDADES CONEXAS',                                                                                                
        'PROGRAMACIÓN INFORMÁTICA',                                                                                                                               
        'PROGRAMACIÓN Y TRANSMISIONES DE TELEVISIÓN',                                                                                                            
        'PUBLICIDAD',                                                                                                                                            
        'RECOGIDA DE DESECHOS NO PELIGROSOS',                                                                                                                    
        'RECOGIDA DE DESECHOS PELIGROSOS',                                                                                                                        
        'RECUPERACIÓN DE MATERIALES',                                                                                                                           
        'REPARACIÓN DE APARATOS DE USO DOMÉSTICO Y EQUIPO DOMÉSTICO Y DE JARDINERÍA',                                                                             
        'REPARACIÓN DE APARATOS ELECTRÓNICOS DE CONSUMO',                                                                                                         
        'REPARACIÓN DE EQUIPO DE TRANSPORTE, EXCEPTO VEHÍCULOS AUTOMOTORES',                                                                                      
        'REPARACIÓN DE EQUIPO ELÉCTRICO',                                                                                                                         
        'REPARACIÓN DE EQUIPOS COMUNICACIONALES',                                                                                                                 
        'REPARACIÓN DE MAQUINARIA',                                                                                                                              
        'REPARACIÓN DE ORDENADORES Y EQUIPO PERIFÉRICO',                                                                                                          
        'REPARACIÓN DE OTROS TIPOS DE EQUIPO',                                                                                                                    
        'REPARACIÓN DE PRODUCTOS ELABORADOS DE METAL',                                                                                                            
        'SEGUROS DE VIDA',                                                                                                                                       
        'SEGUROS GENERALES',                                                                                                                                     
        'SUMINISTRO DE COMIDAS POR ENCARGO',                                                                                                                      
        'SUMINISTRO DE VAPOR Y AIRE ACONDICIONADO',                                                                                                              
        'TEJEDURA DE PRODUCTOS TEXTILES',                                                                                                                        
        'TERMINACIÓN Y ACABADO DE EDIFICIOS',                                                                                                                    
        'TRANSMISIONES DE RADIO',                                                                                                                                 
        'TRANSPORTE DE CARGA MARÍTIMO Y DE CABOTAJE',                                                                                                            
        'TRANSPORTE DE CARGA POR CARRETERA',                                                                                                                  
        'TRANSPORTE DE CARGA POR FERROCARRIL',                                                                                                                    
        'TRANSPORTE DE CARGA POR VÍA AÉREA',                                                                                                                     
        'TRANSPORTE DE CARGA, POR VÍAS DE NAVEGACIÓN INTERIORES',                                                                                                
        'TRANSPORTE DE PASAJEROS MARÍTIMO Y DE CABOTAJE',                                                                                                         
        'TRANSPORTE DE PASAJEROS POR VÍA AÉREA',                                                                                                                 
        'TRANSPORTE DE PASAJEROS POR VÍAS DE NAVEGACIÓN INTERIORES',                                                                                              
        'TRANSPORTE INTERURBANO DE PASAJEROS POR FERROCARRIL',                                                                                                    
        'TRANSPORTE URBANO Y SUBURBANO DE PASAJEROS POR VÍA TERRESTRE',                                                                                         
        'TRATAMIENTO Y ELIMINACIÓN DE DESECHOS NO PELIGROSOS',                                                                                                    
        'TRATAMIENTO Y ELIMINACIÓN DE DESECHOS PELIGROSOS',                                                                                                       
        'TRATAMIENTO Y REVESTIMIENTO DE METALES',                                                                                                                
        'VENTA AL POR MAYOR A CAMBIO DE UNA RETRIBUCIÓN O POR CONTRATA',                                                                                         
        'VENTA AL POR MAYOR DE ALIMENTOS, BEBIDAS Y TABACO.',                                                                                                   
        'VENTA AL POR MAYOR DE COMBUSTIBLES SÓLIDOS, LÍQUIDOS Y GASEOSOS Y PRODUCTOS CONEXOS',                                                                  
        'VENTA AL POR MAYOR DE DESPERDICIOS, DESECHOS, CHATARRA Y OTROS PRODUCTOS N.C.P',                                                                        
        'VENTA AL POR MAYOR DE EQUIPO, PARTES Y PIEZAS ELECTRÓNICOS Y DE TELECOMUNICACIONES',                                                                    
        'VENTA AL POR MAYOR DE MAQUINARIA, EQUIPO Y MATERIALES AGROPECUARIOS',                                                                                   
        'VENTA AL POR MAYOR DE MATERIALES DE CONSTRUCCIÓN, ARTÍCULOS DE FERRETERÍA Y EQUIPO Y MATERIALES DE FONTANERÍA Y CALEFACCIÓN.',                         
        'VENTA AL POR MAYOR DE MATERIAS PRIMAS AGROPECUARIAS Y ANIMALES VIVOS.',                                                                                
        'VENTA AL POR MAYOR DE METALES Y MINERALES METALÍFEROS',                                                                                                 
        'VENTA AL POR MAYOR DE ORDENADORES, EQUIPO PERIFÉRICO Y PROGRAMAS DE INFORMÁTICA',                                                                       
        'VENTA AL POR MAYOR DE OTROS ENSERES DOMÉSTICOS',                                                                                                        
        'VENTA AL POR MAYOR DE OTROS TIPOS DE MAQUINARIA Y EQUIPO',                                                                                             
        'VENTA AL POR MAYOR DE PRODUCTOS TEXTILES, PRENDAS DE VESTIR Y CALZADO',                                                                                 
        'VENTA AL POR MAYOR NO ESPECIALIZADA',                                                                                                                  
        'VENTA AL POR MENOR DE ALIMENTOS EN COMERCIOS ESPECIALIZADOS',                                                                                          
        'VENTA AL POR MENOR DE ALIMENTOS, BEBIDAS Y TABACO EN PUESTOS DE VENTA Y MERCADOS',                                                                      
        'VENTA AL POR MENOR DE APARATOS ELÉCTRICOS DE USO DOMÉSTICO, MUEBLES, EQUIPO DE ILUMINACIÓN Y OTROS ENSERES DOMÉSTICOS EN COMERCIOS ESPECIALIZADOS',     
        'VENTA AL POR MENOR DE ARTÍCULOS DE FERRETERÍA, PINTURAS Y PRODUCTOS DE VIDRIO EN COMERCIOS ESPECIALIZADOS',                                            
        'VENTA AL POR MENOR DE BEBIDAS EN COMERCIOS ESPECIALIZADOS',                                                                                              
        'VENTA AL POR MENOR DE COMBUSTIBLES PARA VEHÍCULOS AUTOMOTORES EN COMERCIOS ESPECIALIZADOS',                                                            
        'VENTA AL POR MENOR DE EQUIPO DE DEPORTE EN COMERCIOS ESPECIALIZADOS',                                                                                    
        'VENTA AL POR MENOR DE EQUIPO DE SONIDO Y DE VÍDEO EN COMERCIOS ESPECIALIZADOS',                                                                          
        'VENTA AL POR MENOR DE LIBROS, PERIÓDICOS Y ARTÍCULOS DE PAPELERÍA EN COMERCIOS ESPECIALIZADOS',                                                         
        'VENTA AL POR MENOR DE ORDENADORES, EQUIPO PERIFÉRICO, PROGRAMAS INFORMÁTICOS Y EQUIPO DE TELECOMUNICACIONES EN COMERCIOS ESPECIALIZADOS',               
        'VENTA AL POR MENOR DE OTROS PRODUCTOS EN PUESTOS DE VENTA Y MERCADOS',                                                                                  
        'VENTA AL POR MENOR DE OTROS PRODUCTOS NUEVOS EN COMERCIOS ESPECIALIZADOS',                                                                             
        'VENTA AL POR MENOR DE PRENDAS DE VESTIR, CALZADO Y ARTÍCULOS DE CUERO EN COMERCIOS ESPECIALIZADOS',                                                     
        'VENTA AL POR MENOR DE PRODUCTOS FARMACÉUTICOS Y MEDICINALES, COSMÉTICOS Y ARTÍCULOS DE TOCADOR EN COMERCIOS ESPECIALIZADOS',                            
        'VENTA AL POR MENOR DE PRODUCTOS TEXTILES EN COMERCIOS ESPECIALIZADOS',                                                                                   
        'VENTA AL POR MENOR DE PRODUCTOS TEXTILES, PRENDAS DE VESTIR Y CALZADO EN PUESTOS DE VENTA Y MERCADOS',                                                   
        'VENTA AL POR MENOR EN COMERCIOS NO ESPECIALIZADOS CON PREDOMINIO DE LA VENTA DE ALIMENTOS, BEBIDAS O TABACO',                                           
        'VENTA AL POR MENOR POR CORREO Y POR INTERNET',                                                                                                           
        #'VENTA DE VEHÍCULOS AUTOMOTORES'                                                                                                                     128
        #'VENTA, MANTENIMIENTO Y REPARACIÓN DE MOTOCICLETAS Y DE SUS PARTES, PIEZAS Y ACCESORIOS.',                                                               
        #'VENTAS DE PARTES, PIEZAS Y ACCESORIOS PARA VEHÍCULOS AUTOMOTORES'
         ])
        Departament = st.sidebar.selectbox('Nombre del Departamento', ['AMAZONAS','AREQUIPA','ÁNCASH','APURÍMAC','AYACUCHO','HUANCAVELICA','HUÁNUCO','JUNÍN','MADRE DE DIOS','MOQUEGUA','PASCO','SAN MARTÍN','TACNA','TUMBES','UCAYALI','PUNO','LIMA','CALLAO','CUSCO','LA LIBERTAD','JUNÍN','CAJAMARCA','LAMBAYEQUE','LORETO'])
        Tama = st.sidebar.selectbox('Tamaño de Empresa', ['MICRO', 'PEQUEÑA','MEDIANA','GRANDE'])
        st.sidebar.header('Medida de seguridad: Si(1), No(0)') 
        F1 = st.sidebar.slider('Infraestructura física (alambrado, muros, etc.?', 0,1)
        F2 = st.sidebar.slider('Sistema de video y captura de imágenes?', 0,1)
        F3 = st.sidebar.slider('Sistema de control de acceso de personal?', 0,1)
        F4 = st.sidebar.slider('Sistema de alarma de seguridad electrónica?', 0,1)
        F5 = st.sidebar.slider('Seguridad para el traslado de valores?', 0,1)
        F6 = st.sidebar.slider('Seguridad para el traslado de bienes?', 0,1)
        F7 = st.sidebar.slider('Personal para resguardo (guardaespaldas)?',0,1)
        F8 = st.sidebar.slider('Personal de seguridad de bienes e inmuebles?', 0,1)
        
        features  = {'acti2': acti2	,
            'Departament': Departament,
            'Tama': Tama,
            'F1': F1,
            'F2': F2,
            'F3': F3,
            'F4': F4,
            'F5': F5,
            'F6': F6,
            'F7': F7,
            'F8': F8}
        data = pd.DataFrame(features,index=[0])

        return data
    st.set_option('deprecation.showPyplotGlobalUse', False)

    user_input_df = get_user_input()
    processed_user_input = data_preprocessor(user_input_df)

    st.subheader('**Parámetros de entrada de usuario**')
    st.write(user_input_df)

    st.subheader('**Parámetros de Predicción**')
    prediction = model.predict(processed_user_input)
    prediction_proba = model.predict_proba(processed_user_input)

    visualize_confidence_level(prediction_proba)

    features_df  = pd.DataFrame(user_input_df)
############################

###########################

  
#if st.button('Predicción'):
    
    prediction = predict_quality(model, features_df)
    
    #st.write('Según sus selecciones, el modelo predice un valor de '+ str(prediction))
    #st.subheader('**Parámetros de Recomendación**')

    if prediction == 'Rob':
        st.subheader("Según sus selecciones, el modelo predice un valor: Robo")
        st.text("Mobiliario, Maquinaria o equipo industrial, Equipo electrónico, Mercancia por parte del personal, Mercancia por parte de los clientes, Dinero, tarjetas de crédito o cheques, Vehículos")##centari
        #img = Image.open("images/5.jpg")
        #st.image(img, width=300)#captin
       
    elif prediction == 'Int':
        st.subheader("Según sus selecciones, el modelo predice un valor: Intento")##centari
        #img = Image.open("images/4.jpg")#Iagen
        #st.image(img, width=300)#captin
       
    elif prediction == 'Est':
        st.subheader("Según sus selecciones, el modelo predice un valor: Estafa")##centari
        st.text("Pago o prestación de un producto y/o servicio no retribuido (por el cliente o proveedor), Cheque o dinero falso,Desvío de recursos por personal de la empresa, Con tarjeta de débito o crédito, Por internet / correo electrónico ")##centari
        #img = Image.open("images/8.jpg")#Iagen
        #st.image(img, width=300)#captin
      
    elif prediction == 'Rob_Est':
        st.subheader("Según sus selecciones, el modelo predice un valor: Robo y Estafa")##centari
        #img = Image.open("images/1.jpg")#Iagen
        #st.image(img, width=300)#captin
       
    elif prediction == 'Rob_Int':
        st.subheader("Según sus selecciones, el modelo predice un valor: Robo e intent")##centari
        #img = Image.open("images/7.jpg")#Iagen
        #st.image(img, width=300)#captin
       
    elif prediction == 'Int_Est':
        st.subheader("Según sus selecciones, el modelo predice un valor: Intento y estafa")##centari
        #img = Image.open("images/6.jpg")#Iagen
        #st.image(img, width=300)#captin
      
    elif prediction == 'Rob_Int_Est':
        st.subheader("Según sus selecciones, el modelo predice un valor: Rob Intento Estafa")##centari
        #img = Image.open("images/9.jpg")#Iagen
        #st.image(img, width=300)#captin
        
    st.success("Te recordamos evitar actividades rutinarias, exposicines y lugares peligrosos.") #brde






  