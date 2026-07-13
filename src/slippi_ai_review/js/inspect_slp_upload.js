"use strict";

function fail(message) {
  process.stderr.write(`${message}\n`);
  process.exit(2);
}

try {
  const replayPath = require("path").resolve(process.argv[2] || "");
  const slippi = require("@slippi/slippi-js/node");
  const game = new slippi.SlippiGame(replayPath);
  const settings = game.getSettings();
  const latestFrame = game.getLatestFrame();
  const metadata = game.getMetadata();
  if (!settings || !latestFrame || !Array.isArray(settings.players)) {
    fail("Missing Slippi settings or frame data");
  }

  const players = settings.players.map((player) => {
    const characterInfo = slippi.characters.getCharacterInfo(player.characterId);
    if (!characterInfo || !characterInfo.name) fail("Unknown character identity");
    const colorId = Number.isInteger(player.characterColor) ? player.characterColor : 0;
    const colorName = slippi.characters.getCharacterColorName(player.characterId, colorId);
    const framePlayer = latestFrame.players?.[player.playerIndex];
    if (!framePlayer?.post) fail("Missing player frame data");
    return {
      playerIndex: player.playerIndex,
      port: player.port,
      name: player.displayName || player.nametag || null,
      displayName: player.displayName || null,
      nametag: player.nametag || null,
      connectCode: player.connectCode || null,
      character: { id: player.characterId, name: characterInfo.name, colorId, colorName },
    };
  });

  process.stdout.write(JSON.stringify({
    players,
    match: {
      stageId: settings.stageId,
      slpVersion: settings.slpVersion,
      lastFrame: metadata?.lastFrame ?? latestFrame.frame,
    },
  }));
} catch (error) {
  fail(error instanceof Error ? error.message : "Unreadable Slippi replay");
}
