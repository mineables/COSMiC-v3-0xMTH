const Miner = require("./0xbitcoinminer-accel");
const Vault = require("./lib/vault");
const miningLogger = require("./lib/mining-logger");
var prompt = require('prompt');
var pjson = require('./package.json');
var Web3 = require('web3')
var NetworkInterface = require("./lib/network-interface");
var PoolInterface = require("./lib/pool-interface");
var ArtifactInterface = require("./lib/artifact-interface");

var web3 = new Web3();
var running = true;

init()

async function init() {
    initSignalHandlers();

    prompt.message = null;
    prompt.delimiter = ":";
    prompt.start();

    if(process.argv.length > 2) {
        handleCommand(process.argv.slice(2).join(' '));
        if(process.argv[2] == 'help') {
            return process.exit();
        } else {
            drawLayout();
        }
    } else {
        drawLayout();
        console.log('Welcome to COSMiC: Community Open-Source Miner including CUDA [\x1b[38;5;86mMineables Edition\x1b[0m]')
        console.log('HashBurner build by LtTofu [V3.4t] - "Cookin your Hashbrowns" ')
        console.log('based on v2.10.0pre5+ by Mikers, Azlehria & the 0xBitcoin Discord Crew')
        console.log('* Experimental Build for Older CUDA Devices (Kepler and previous) *')
        console.log('')
        console.log('Type a command to get started.  Type "help" for a list of commands.')
    }
    // explicitly ask for password
    await Vault.assertValidPassword()
    return getPrompt();
}

async function getPrompt() {
    var result = await promptForCommand();

    return getPrompt();
}

function sigHandler(signal) {
    process.exit(128 + signal)
}

function initSignalHandlers() {
    process.on('SIGTERM', sigHandler);
    process.on('SIGINT', sigHandler);
    process.on('SIGBREAK', sigHandler);
    process.on('SIGHUP', sigHandler);
    process.on('SIGWINCH', (sig) => {
        process.stdout.write("\x1b[5r\x1b[5;1f");
    });
    process.on('exit', (sig) => {
        process.stdout.write("\x1b[s\x1b[?25h\x1b[r\x1b[u");
    });
}

function drawLayout() {
    process.stdout.write( "\x1b[?25l\x1b[2J\x1b(0" );
    process.stdout.write( "\x1b[1;1flqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqwqqqqqqqqqqqqqqqqqqqqqqqqqqwqqqqqqqqqqqqqqqqqk" );
    process.stdout.write( "\x1b[4;1fmqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqvqqqqqqqqqqqqqqqqqqqqqqqqqqvqqqqqqqqqqqqqqqqqj" );
    process.stdout.write( "\x1b[2;1fx\x1b[2;35fx\x1b[2;62fx\x1b[2;80fx" );
    process.stdout.write( "\x1b[3;1fx\x1b[3;35fx\x1b[3;62fx\x1b[3;80fx" );
    process.stdout.write( "\x1b(B\x1b[2;2fChallenge:" );
    process.stdout.write( "\x1b[3;2fDifficulty:" );
    process.stdout.write( "\x1b[2;37fHashes This Round" );
    process.stdout.write( "\x1b[2;63fRound time:" );
    process.stdout.write( "\x1b[3;63fAccount:" );
    process.stdout.write( "\x1b[2;31fMH/s" );
    process.stdout.write( "\x1b[3;31fSols" );
    process.stdout.write( "\x1b[s\x1b[3;29f\x1b[38;5;221m0\x1b[0m\x1b[u" );
    process.stdout.write( "\x1b[1;64f" + pjson.version );
    process.stdout.write( "\x1b[5r\x1b[5;1f\x1b[?25h" );
}

async function promptForCommand() {
    return new Promise(function (fulfilled, rejected) {
        prompt.get(['command'], async function (err, result) {
            if (err) {
                console.log(err);
                return rejected(err);
            } else {
                var response = await handleCommand(result.command)
                return fulfilled(response);
            }
        });
    });
}

async function handleCommand(result) {
    var split_command = result.split(' ');
    //console.log( split_command )

    var subsystem_name = split_command[0];
    var subsystem_command = split_command[1];
    var subsystem_option = split_command[2];

    if (subsystem_name == 'login' ) {
	
	await Vault.assertValidPassword()
    }

    if (subsystem_name == 'account') {
        if (subsystem_command === 'new' || subsystem_command === 'list') {
            Vault.requirePassword(true) //for encryption of private key !
        }

        var unlocked = await Vault.init(web3, miningLogger);
        if (!unlocked) return false;

        await Vault.handleAccountCommand(subsystem_command, subsystem_option)
    }

    if (subsystem_name == 'contract') {
        var unlocked = await Vault.init(web3, miningLogger);
        if (!unlocked) return false;

        await Vault.handleContractCommand(subsystem_command, subsystem_option)
    }

    if (subsystem_name == 'config') {
        var unlocked = await Vault.init(web3, miningLogger);
        if (!unlocked) return false;

        await Vault.handleConfigCommand(subsystem_command, subsystem_option)
    }

    if (subsystem_name == 'mine') {
        Vault.requirePassword(true) //for encryption of private key !

        var unlocked = await Vault.init(web3, miningLogger);
        if (!unlocked) return false;

        NetworkInterface.init(web3, Vault, miningLogger);

        Miner.init(web3, Vault, miningLogger);
        Miner.setNetworkInterface(NetworkInterface);

        Miner.setMiningStyle("solo")

        //us command as option -- for cuda or opencl
        subsystem_option = subsystem_command;
        Miner.mine(subsystem_command, subsystem_option)
    }

    if (subsystem_name == 'pool') {
        Vault.requirePassword(true) //for encryption of private key !
        var unlocked = await Vault.init(web3, miningLogger);
        if (!unlocked) return false;

        await PoolInterface.init(web3, subsystem_command, Vault, miningLogger);
        await PoolInterface.handlePoolCommand(subsystem_command, subsystem_option)

        if (subsystem_command == "mine") {
            Miner.init(web3, Vault, miningLogger);
            Miner.setNetworkInterface(PoolInterface);
            Miner.setMiningStyle("pool")
            Miner.mine(subsystem_command, subsystem_option)
        }
    }

    if (subsystem_name == 'exit' || subsystem_name == 'quit') {
        process.exit(0);
    }

    if (subsystem_name == 'help') {
        return printHelp();
    }
}

function printHelp() {
    console.log('--Mineables Miner Help--')
    console.log('Available commands:\n')

    console.log('"account new"            - Create a new account and local keystore (.0xmithril)')
    console.log('"account list"           - List accounts (local keystore in .0xmithril or specified address')
    console.log('"account remove [index]" - Remove account at a certain index')
    console.log('"account select 0x####"  - Select the active mining account by address')
    console.log('"account balance"        - List the Ether & Token balance of the active account')
    console.log('"account import 0x####"  - Import an existing account with private key\n')
    
    console.log('"contract list"          - List the selected token contract to mine')
    console.log('"contract select 0x####" - Select a PoW token contract to mine\n')

    console.log('"config list"            - Show your current configuration')
    console.log('"config vardiff #"      - Set custom vardiff for pool mining')
    console.log('"config gasprice #"      - Set the gasprice used to submit PoW in solo mining')
    //  console.log('"config cpu_threads #"   - Set the number of CPU cores to use for mining ')
    console.log('"config web3provider http://----:####" - Set the web3 provider \n')

    console.log('"pool mine"              - Begin mining into a pool using CPU')
    console.log('"pool mine cuda"         - Begin mining into a pool using CUDA GPU')
    //console.log('"pool mine opencl"       - Begin mining into a pool using OpenCL GPU')
    console.log('"pool list"              - List the selected mining pool')
    console.log('"pool select http://####.com:####" - Select a pool to mine into\n')

    console.log('"mine"                   - Begin solo mining')
    console.log('"mine cuda"              - Begin solo mining using CUDA GPU')
    //console.log('"mine opencl" - Begin mining using OpenCL GPU')
    //  console.log('Encrypted data vault stored at '+ Vault.get0xBitcoinLocalFolderPath())
}
