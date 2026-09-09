const input=document.getElementById("fileInput");
const drop=document.getElementById("dropZone");
const info=document.getElementById("fileInfo");
const nameEl=document.getElementById("fileName");
const remove=document.getElementById("removeButton");
const workspace=document.getElementById("workspace");
const original=document.getElementById("originalPreview");
const result=document.getElementById("resultPreview");
const placeholder=document.getElementById("placeholder");
const button=document.getElementById("colorizeButton");
const spinner=document.getElementById("spinner");
const download=document.getElementById("downloadButton");
const status=document.getElementById("status");
let selected=null;
function msg(text,error=false){status.textContent=text;status.className=error?"error":"";}
function choose(file){
 if(!file)return;
 if(!["image/jpeg","image/png","image/jpg"].includes(file.type)){msg("Please choose JPG, JPEG or PNG.",true);return;}
 if(file.size>15*1024*1024){msg("Maximum size is 15 MB.",true);return;}
 selected=file;nameEl.textContent=file.name;info.classList.remove("hidden");workspace.classList.remove("hidden");
 original.src=URL.createObjectURL(file);result.classList.add("hidden");placeholder.classList.remove("hidden");download.classList.add("hidden");button.disabled=false;msg("Image ready.");
}
drop.onclick=()=>input.click();
drop.ondragover=e=>{e.preventDefault();drop.classList.add("drag");};
drop.ondragleave=()=>drop.classList.remove("drag");
drop.ondrop=e=>{e.preventDefault();drop.classList.remove("drag");choose(e.dataTransfer.files[0]);};
input.onchange=()=>choose(input.files[0]);
remove.onclick=()=>{selected=null;input.value="";info.classList.add("hidden");workspace.classList.add("hidden");button.disabled=true;download.classList.add("hidden");msg("");};
button.onclick=async()=>{
 if(!selected)return;
 button.disabled=true;spinner.classList.remove("hidden");msg("Running DeOldify inference...");
 const fd=new FormData();fd.append("file",selected);
 try{const r=await fetch("/api/colorize",{method:"POST",body:fd});const d=await r.json();if(!r.ok)throw new Error(d.detail||"Colorization failed.");const url=d.result_url+"?t="+Date.now();result.src=url;result.classList.remove("hidden");placeholder.classList.add("hidden");download.href=url;download.classList.remove("hidden");msg("Colorization complete.");}
 catch(e){msg(e.message,true);}finally{button.disabled=false;spinner.classList.add("hidden");}
};
