"""Run with Python 3.11 after SETUP_WINDOWS.bat. Webcam inference is local."""
from pathlib import Path
import queue
import math
import threading
import time
import tkinter as tk
from tkinter import ttk

ROOT = Path(__file__).resolve().parent
MODEL_ID = 'dima806/asl_alphabet_image_detection'


def display_label(label, confidence, threshold=.70):
    """Only show supported static alphabet labels above the selected threshold."""
    label = label.strip().upper()
    if label in ('J', 'Z'):
        return 'Motion needed'
    if len(label) != 1 or not ('A' <= label <= 'Z') or confidence < threshold:
        return 'Uncertain'
    return label


def ly_handshape(points):
    """Conservative L/Y geometry check. Points use equal units in x/y/z.

    Returns None for ambiguous, folded-thumb, edge-on, or degenerate hands.
    This is a handshape heuristic, not a calibrated model probability.
    """
    if len(points) != 21:
        return None
    def sub(a,b): return tuple(x-y for x,y in zip(a,b))
    def norm(v): return math.sqrt(sum(x*x for x in v))
    def dist(a,b): return norm(sub(points[a],points[b]))
    def angle(a,b,c):
        u,v=sub(points[a],points[b]),sub(points[c],points[b])
        length=norm(u)*norm(v)
        if length < 1e-10: return 0.
        cosine=max(-1.,min(1.,sum(x*y for x,y in zip(u,v))/length))
        return math.degrees(math.acos(cosine))
    palm=dist(5,17)
    if palm < 1e-8: return None
    u,v=sub(points[5],points[0]),sub(points[17],points[0])
    normal=(u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0])
    if norm(normal) < 1e-10 or abs(normal[2])/norm(normal) < .45:
        return None
    # Thumb straight and away from the index base; scale invariant.
    if angle(2,3,4) < 145 or dist(4,5) < .65*palm:
        return None
    states=[]
    for base,pip,dip,tip in [(5,6,7,8),(9,10,11,12),(13,14,15,16),(17,18,19,20)]:
        joint=angle(base,pip,dip)
        if joint >= 155 and angle(pip,dip,tip)>=150 and dist(0,tip)>1.12*dist(0,pip):
            states.append('open')
        elif joint <= 130 and dist(0,tip)<1.05*dist(0,pip):
            states.append('closed')
        else:
            states.append('unknown')
    if states == ['open','closed','closed','closed']: return 'L'
    if states == ['closed','closed','closed','open']: return 'Y'
    return None


def prediction_text(label,confidence,threshold,stable_shape):
    # Limit intervention to the observed L/Y confusion. Never relabel the
    # model's L confidence as confidence in a geometry-corrected Y.
    raw=label.strip().upper()
    if raw in ('L','Y'):
        if stable_shape in ('L','Y'):
            return f'{stable_shape} | hand shape'
        return 'Checking L / Y'
    return f'{display_label(label,confidence,threshold)} {confidence:.0%}'


class LiveApp:
    def __init__(self, root):
        self.root = root
        root.title('ASL Live | Alphabet Recognition')
        root.geometry('1280x850')
        root.minsize(960,720)
        root.configure(bg='#151910')
        self.stop_event = threading.Event()
        self.events = queue.Queue()
        self.frames = queue.Queue(maxsize=1)
        self.worker = None
        self.camera_index = tk.StringVar(value='0')
        self.threshold = tk.DoubleVar(value=.70)
        self.status = tk.StringVar(value='Ready when you are. Click Start camera to begin.')
        self.result = tk.StringVar(value='Waiting for camera')
        self.letter = tk.StringVar(value='—')
        self.detail = tk.StringVar(value='Your detected letter will appear here')
        self.session_state = tk.StringVar(value='STANDBY')
        self.processor = self.model = None
        self.closing = False
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame',background='#151910')
        style.configure('TLabel',background='#151910',foreground='#f3efdf',font=('Segoe UI',11))
        style.configure('Start.TButton',background='#c1cd8b',foreground='#252d17',
                        font=('Segoe UI',11,'bold'),padding=(20,12),borderwidth=0)
        style.map('Start.TButton',background=[('disabled','#424a30'),('active','#d4dfa7')])
        style.configure('Stop.TButton',background='#493e2d',foreground='#ecdbb8',
                        font=('Segoe UI',11,'bold'),padding=(20,12),borderwidth=0)
        style.map('Stop.TButton',background=[('disabled','#2d3224'),('active','#625039')])
        style.configure('TEntry',fieldbackground='#232a1c',foreground='#f3efdf',padding=7)
        style.configure('TSpinbox',fieldbackground='#232a1c',foreground='#f3efdf',padding=7,
                        arrowsize=14)
        def label(parent,text='',size=11,color='#f3efdf',bold=False,**kw):
            return tk.Label(parent,text=text,bg=parent.cget('bg'),fg=color,
                            font=('Segoe UI',size,'bold' if bold else 'normal'),**kw)
        header=tk.Frame(root,bg='#151910'); header.pack(fill='x',padx=26,pady=(22,16))
        label(header,'ASL Live',26,bold=True).pack(side='left')
        label(header,'American Sign Language · Alphabet recognition',11,'#a9ae94').pack(side='left',padx=20)
        label(header,textvariable=self.session_state,size=10,color='#c1cd8b',bold=True).pack(side='right')
        controls=tk.Frame(root,bg='#252d1e',highlightbackground='#424b32',highlightthickness=1)
        controls.pack(fill='x',padx=26,pady=(0,20))
        row=tk.Frame(controls,bg='#252d1e'); row.pack(fill='x',padx=16,pady=12)
        self.start_button=ttk.Button(row,text='Start camera',style='Start.TButton',command=self.start)
        self.start_button.pack(side='left')
        self.stop_button=ttk.Button(row,text='Stop',style='Stop.TButton',command=self.stop,state='disabled')
        self.stop_button.pack(side='left',padx=(10,26))
        label(row,'Camera',10,'#bdc1a9').pack(side='left',padx=(0,8))
        self.camera_input=ttk.Entry(row,textvariable=self.camera_index,width=4)
        self.camera_input.pack(side='left')
        label(row,'Confidence threshold',10,'#bdc1a9').pack(side='left',padx=(24,8))
        self.threshold_input=ttk.Spinbox(row,from_=.5,to=.99,increment=.05,textvariable=self.threshold,width=5)
        self.threshold_input.pack(side='left')
        body=tk.Frame(root,bg='#151910'); body.pack(fill='both',expand=True,padx=26)
        body.columnconfigure(0,weight=4,uniform='panels')
        body.columnconfigure(1,weight=7,uniform='panels')
        body.rowconfigure(0,weight=1)
        left=tk.Frame(body,bg='#252d1e',highlightbackground='#424b32',highlightthickness=1)
        left.grid(row=0,column=0,sticky='nsew',padx=(0,16))
        label(left,'Alphabet guide',16,bold=True).pack(anchor='w',padx=18,pady=(16,4))
        label(left,'Find a letter, then match the hand shape.',10,'#a9ae94').pack(anchor='w',padx=18,pady=(0,12))
        self.reference=tk.Canvas(left,bg='#252d1e',highlightthickness=0)
        self.reference.pack(fill='both',expand=True,padx=12,pady=(0,12))
        self.reference.bind('<Configure>',self.draw_reference)
        right=tk.Frame(body,bg='#151910'); right.grid(row=0,column=1,sticky='nsew')
        video_panel=tk.Frame(right,bg='#252d1e',highlightbackground='#424b32',highlightthickness=1)
        video_panel.pack(fill='both',expand=True)
        label(video_panel,'Live camera',16,bold=True).pack(anchor='w',padx=18,pady=(16,4))
        label(video_panel,'Keep one hand fully visible and hold your sign.',10,'#a9ae94').pack(anchor='w',padx=18,pady=(0,12))
        self.video=tk.Canvas(video_panel,bg='#1b2116',highlightthickness=0,height=300)
        self.video.pack(fill='both',expand=True,padx=12,pady=(0,12))
        self.video.bind('<Configure>',self.camera_placeholder)
        result_card=tk.Frame(right,bg='#303a24',highlightbackground='#626d43',highlightthickness=1)
        result_card.pack(fill='x',pady=(16,0))
        label(result_card,textvariable=self.letter,size=56,color='#d0dca0',bold=True,width=2).pack(side='left',padx=(18,16),pady=5)
        result_copy=tk.Frame(result_card,bg='#303a24'); result_copy.pack(side='left',fill='x',expand=True,pady=16)
        label(result_copy,'DETECTED LETTER',10,'#c3c7ac',bold=True).pack(anchor='w')
        self.detail_label=label(result_copy,textvariable=self.detail,size=12,wraplength=380,justify='left')
        self.detail_label.pack(anchor='w',pady=(7,0))
        footer=tk.Frame(root,bg='#151910'); footer.pack(fill='x',padx=26,pady=(14,18))
        label(footer,'STATUS',9,'#c1cd8b',bold=True).pack(side='left',padx=(0,12))
        self.status_label=label(footer,textvariable=self.status,size=10,color='#a9ae94',wraplength=1000,justify='left')
        self.status_label.pack(side='left',fill='x',expand=True)
        self.result.trace_add('write',self.update_result)
        root.protocol('WM_DELETE_WINDOW',self.close)
        root.after(60,self.poll)

    def update_result(self,*args):
        text=self.result.get()
        parts=text.split()
        recognized=bool(parts and len(parts[0])==1 and 'A'<=parts[0]<='Z')
        self.letter.set(parts[0] if recognized else '—')
        if recognized:
            self.detail.set('Hand-shape check' if 'hand shape' in text else 'Model confidence: '+parts[-1])
        else:
            self.detail.set({'Waiting for camera':'Your detected letter will appear here',
                             'No hand detected':'Place your hand inside the camera view',
                             'Stopped':'Camera stopped. Start again when ready.',
                             'Loading...':'Getting the camera and model ready...'}.get(text,text))

    def camera_placeholder(self,event=None):
        if self.worker and self.worker.is_alive(): return
        self.video.delete('all')
        x,y=self.video.winfo_width()/2,self.video.winfo_height()/2
        self.video.create_text(x,y-14,text='Your camera preview',fill='#f0eddc',font=('Segoe UI',19,'bold'))
        self.video.create_text(x,y+20,text='Click Start camera to begin',fill='#a4ad8c',font=('Segoe UI',11))

    def draw_reference(self,event=None):
        from PIL import Image, ImageTk
        img = Image.open(ROOT/'assets'/'reference-000.jpg')
        img.thumbnail((max(1,self.reference.winfo_width()-10),max(1,self.reference.winfo_height()-10)))
        self.ref_photo = ImageTk.PhotoImage(img)
        self.reference.delete('all')
        self.reference.create_image(self.reference.winfo_width()/2,self.reference.winfo_height()/2,image=self.ref_photo)

    def start(self):
        if self.worker and self.worker.is_alive(): return
        try:
            index = int(self.camera_index.get())
            if index < 0: raise ValueError()
            threshold = float(self.threshold.get())
            if not .0 < threshold <= 1.: raise ValueError()
        except (ValueError,tk.TclError):
            self.status.set('Enter a camera number (0, 1...) and confidence between 0 and 1.'); return
        self.stop_event.clear()
        self.start_button.configure(state='disabled')
        self.stop_button.configure(state='normal')
        self.camera_input.configure(state='disabled')
        self.threshold_input.configure(state='disabled')
        self.session_state.set('STARTING')
        self.result.set('Loading...')
        self.worker = threading.Thread(target=self.run,args=(index,threshold),daemon=True)
        self.worker.start()

    def run(self,index,threshold):
        cap = hands = None
        try:
            import cv2
            import mediapipe as mp
            import torch
            from PIL import Image
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            if self.model is None:
                self.events.put(('status','Downloading/loading pretrained model. Keep this window open; see VS Code terminal for progress.'))
                torch.set_num_threads(2)
                self.processor = AutoImageProcessor.from_pretrained(MODEL_ID,use_fast=False)
                self.model = AutoModelForImageClassification.from_pretrained(MODEL_ID,use_safetensors=True,trust_remote_code=False)
                self.model.eval()
            if self.stop_event.is_set(): return
            # Try Windows camera backends in order, validating an actual frame.
            import sys
            backends = [cv2.CAP_DSHOW,cv2.CAP_MSMF,cv2.CAP_ANY] if sys.platform=='win32' else [cv2.CAP_ANY]
            initial = None
            for backend in backends:
                candidate = cv2.VideoCapture(index,backend)
                if candidate.isOpened():
                    candidate.set(cv2.CAP_PROP_FRAME_WIDTH,640)
                    candidate.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
                    for _ in range(8):
                        ok, frame = candidate.read()
                        if ok and frame is not None:
                            cap, initial = candidate, frame; break
                if cap is not None: break
                candidate.release()
            if cap is None:
                raise RuntimeError('Cannot open camera. Close other camera apps, then try camera 0 or 1.')
            hands = mp.solutions.hands.Hands(max_num_hands=1,min_detection_confidence=.6,min_tracking_confidence=.6)
            self.events.put(('status','Ready. Hold one static ASL sign with your hand fully visible. Predictions can be wrong.'))
            last_inference = 0.
            label, confidence, accepted = '',0.,'Uncertain'
            shape_candidate, shape_since = None, 0.
            box_old = None
            while not self.stop_event.is_set():
                if initial is not None:
                    frame, initial = initial, None
                else:
                    ok,frame = cap.read()
                    if not ok: raise RuntimeError('Camera stopped providing frames. Click Stop then Start.')
                frame = cv2.flip(frame,1)
                rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                detection = hands.process(rgb)
                if detection.multi_hand_landmarks:
                    hand = detection.multi_hand_landmarks[0]
                    h,w = frame.shape[:2]
                    points=[(p.x*w,p.y*h,p.z*w) for p in hand.landmark]
                    shape=ly_handshape(points)
                    shape_now=time.monotonic()
                    if shape != shape_candidate:
                        shape_candidate,shape_since=shape,shape_now
                    stable_shape=shape if shape and shape_now-shape_since>=.25 else None
                    xs,ys = [p.x*w for p in hand.landmark],[p.y*h for p in hand.landmark]
                    margin = int(max(max(xs)-min(xs),max(ys)-min(ys))*.20)+8
                    x1,y1 = max(0,int(min(xs))-margin),max(0,int(min(ys))-margin)
                    x2,y2 = min(w,int(max(xs))+margin),min(h,int(max(ys))+margin)
                    now = time.monotonic()
                    box = (x1,y1,x2,y2)
                    # Invalidate the old result if the hand jumps to a different location.
                    if box_old and max(abs(a-b) for a,b in zip(box,box_old)) > 90:
                        last_inference=0
                        accepted='Uncertain'
                    box_old = box
                    if now-last_inference >= .30 and x2>x1 and y2>y1:
                        inputs = self.processor(images=Image.fromarray(rgb[y1:y2,x1:x2]),return_tensors='pt')
                        with torch.inference_mode():
                            probs = self.model(**inputs).logits.softmax(-1)[0]
                        idx = int(probs.argmax().item())
                        label = self.model.config.id2label[idx]
                        confidence = float(probs[idx].item())
                        accepted = display_label(label,confidence,threshold)
                        last_inference=now
                    # Crop and inference happen before adding UI drawings.
                    color=(125,155,165)
                    cv2.rectangle(frame,(x1,y1),(x2-1,y2-1),color,3)
                    text = prediction_text(label,confidence,threshold,stable_shape)
                    (tw,th),_ = cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,.7,2)
                    tx=min(x1,max(0,w-tw-16)); ty=max(th+14,y1)
                    cv2.rectangle(frame,(tx,ty-th-12),(min(w-1,tx+tw+12),ty+4),color,-1)
                    cv2.putText(frame,text,(tx+5,ty-3),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
                    self.events.put(('result',text))
                else:
                    label,confidence,accepted='',0.,'Uncertain'
                    last_inference=0.; box_old=None
                    shape_candidate,shape_since=None,0.
                    self.events.put(('result','No hand detected'))
                image = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                try: self.frames.get_nowait()
                except queue.Empty: pass
                self.frames.put(image)
        except Exception as exc:
            self.events.put(('status',f'Could not start: {exc}. Check the terminal and README_AR.md.'))
            self.events.put(('result','Not running'))
        finally:
            if cap is not None: cap.release()
            if hands is not None: hands.close()
            self.events.put(('done',None))

    def stop(self):
        self.stop_event.set()
        self.stop_button.configure(state='disabled')
        self.session_state.set('STOPPING')
        self.status.set('Stopping camera. Please wait...')

    def poll(self):
        if self.closing: return
        from PIL import ImageTk
        # Only the GUI thread accesses Tk widgets.
        latest={}
        while True:
            try:
                kind,value=self.events.get_nowait(); latest[kind]=value
            except queue.Empty: break
        if 'status' in latest:
            self.status.set(latest['status'])
            if latest['status'].startswith('Ready.'):
                self.session_state.set('LIVE')
            elif latest['status'].startswith('Could not start:'):
                self.session_state.set('NEEDS ATTENTION')
        if 'result' in latest: self.result.set(latest['result'])
        if 'done' in latest:
            self.start_button.configure(state='normal')
            self.stop_button.configure(state='disabled')
            self.camera_input.configure(state='normal')
            self.threshold_input.configure(state='normal')
            if self.stop_event.is_set():
                self.session_state.set('STANDBY')
                self.result.set('Stopped'); self.status.set('Stopped. Click Start to resume.')
                self.camera_placeholder()
                try: self.frames.get_nowait()
                except queue.Empty: pass
        try:
            frame=self.frames.get_nowait()
            frame.thumbnail((max(1,self.video.winfo_width()-8),max(1,self.video.winfo_height()-8)))
            self.frame_photo=ImageTk.PhotoImage(frame)
            self.video.delete('all')
            self.video.create_image(self.video.winfo_width()/2,self.video.winfo_height()/2,image=self.frame_photo)
        except queue.Empty: pass
        self.root.after(60,self.poll)

    def close(self):
        self.closing=True
        self.stop_event.set()
        self.root.destroy()


if __name__ == '__main__':
    root=tk.Tk()
    LiveApp(root)
    root.mainloop()
