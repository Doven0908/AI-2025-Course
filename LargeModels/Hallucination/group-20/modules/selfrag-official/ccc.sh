#!/bin/bash
# Claude Code Companion - Linux Launcher
echo "Configuring Claude Code for proxy use..."

# Set environment variables first (fallback and primary method)
export ANTHROPIC_BASE_URL="http://115.190.196.178:8080"
export ANTHROPIC_AUTH_TOKEN="hello"
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1"
export API_TIMEOUT_MS="600000"

# Use embedded Node.js to process settings.json if it exists
node -e "const fs=require('fs');const path=require('path');const os=require('os');const claudeDir=path.join(os.homedir(),'.claude');const settingsFile=path.join(claudeDir,'settings.json');const targetEnv={'ANTHROPIC_BASE_URL':'http://115.190.196.178:8080','ANTHROPIC_AUTH_TOKEN':'hello','CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC':'1','API_TIMEOUT_MS':'600000'};function processSettings(){if(!fs.existsSync(settingsFile)){console.log('Claude settings file not found, environment variables already set');return true;}try{const content=fs.readFileSync(settingsFile,'utf8');const settings=JSON.parse(content);if(!settings.env)settings.env={};let needsUpdate=false;let backupCreated=false;for(const[key,targetValue]of Object.entries(targetEnv)){const currentValue=settings.env[key];if(currentValue!==targetValue){if(!backupCreated){const timestamp=new Date().toISOString().replace(/[:.]/g,'-');const backupFile=settingsFile+'.backup-'+timestamp;fs.copyFileSync(settingsFile,backupFile);console.log('Backed up settings to: '+backupFile);backupCreated=true;}if(currentValue){console.log('Updating '+key+': '+currentValue+' -> '+targetValue);}else{console.log('Adding '+key+': '+targetValue);}settings.env[key]=targetValue;needsUpdate=true;}}if(needsUpdate){fs.writeFileSync(settingsFile,JSON.stringify(settings,null,2));console.log('Settings updated successfully');}else{console.log('Settings already configured correctly');}return true;}catch(error){console.error('Error processing settings:',error.message);console.log('Environment variables already set as fallback');return false;}}processSettings();" >/dev/null 2>&1

echo "Starting Claude with proxy configuration..."
exec claude "$@"