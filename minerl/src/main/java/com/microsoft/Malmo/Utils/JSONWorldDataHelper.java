// --------------------------------------------------------------------------------------------------
//  Copyright (c) 2016 Microsoft Corporation
//  
//  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
//  associated documentation files (the "Software"), to deal in the Software without restriction,
//  including without limitation the rights to use, copy, modify, merge, publish, distribute,
//  sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//  
//  The above copyright notice and this permission notice shall be included in all copies or
//  substantial portions of the Software.
//  
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
//  NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
//  NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
//  DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
// --------------------------------------------------------------------------------------------------

package com.microsoft.Malmo.Utils;

import com.google.common.base.CaseFormat;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonPrimitive;
import net.minecraft.block.Block;
import net.minecraft.block.BlockState;
import net.minecraft.client.Minecraft;
import net.minecraft.client.entity.player.ClientPlayerEntity;
import net.minecraft.network.play.client.CClientStatusPacket;
import net.minecraft.stats.Stat;
import net.minecraft.stats.StatType;
import net.minecraft.stats.StatisticsManager;
import net.minecraft.stats.Stats;
import net.minecraft.util.RegistryKey;
import net.minecraft.util.ResourceLocation;
import net.minecraft.util.math.BlockPos;
import net.minecraft.util.registry.Registry;
import net.minecraft.world.biome.Biome;

import java.util.Arrays;
import java.util.Map;


/**
 * Helper class for building the "World data" to be passed from Minecraft back to the agent.<br>
 * This class contains helper methods to build up a JSON tree of useful information, such as health, XP, food levels, distance travelled, etc.etc.<br>
 * It can also build up a grid of the block types around the player or somewhere else in the world.
 * Call this on the Server side only.
 */
public class JSONWorldDataHelper
{
    /**
     * Simple class to hold the dimensions of the environment
     * that we want to return in the World Data.<br>
     * Min and max define an inclusive range, where the player's feet are situated at (0,0,0) if absoluteCoords=false.
     */
    static public class GridDimensions {
        public int xMin;
        public int xMax;
        public int yMin;
        public int yMax;
        public int zMin;
        public int zMax;
        public boolean absoluteCoords;
        public boolean projectDown;

        /**
         * Default constructor asks for an environment just big enough to contain
         * the player and one block all around him.
         */
        public GridDimensions() {
            this.xMin = -1; this.xMax = 1;
            this.zMin = -1; this.zMax = 1;
            this.yMin = -1; this.yMax = 2;
            this.absoluteCoords = false;
            this.projectDown = false;
        }

        /**
         * Convenient constructor - effectively specifies the margin around the player<br>
         * Passing (1,1,1) will have the same effect as the default constructor.
         * @param xMargin number of blocks to the left and right of the player
         * @param yMargin number of blocks above and below player
         * @param zMargin number of blocks in front of and behind player
         */
        public GridDimensions(int xMargin, int yMargin, int zMargin) {
            this.xMin = -xMargin; this.xMax = xMargin;
            this.yMin = -yMargin; this.yMax = yMargin + 1;  // +1 because the player is two blocks tall.
            this.zMin = -zMargin; this.zMax = zMargin;
            this.absoluteCoords = false;
            this.projectDown = false;
        }

        /**
         * Convenient constructor for the case where all that is required is the flat patch of ground<br>
         * around the player's feet.
         * @param xMargin number of blocks around the player in the x-axis
         * @param zMargin number of blocks around the player in the z-axis
         */
        public GridDimensions(int xMargin, int zMargin) {
            this.xMin = -xMargin; this.xMax = xMargin;
            this.yMin = -1; this.yMax = -1;  // Flat patch of ground at the player's feet.
            this.zMin = -zMargin; this.zMax = zMargin;
            this.absoluteCoords = false;
            this.projectDown = false;
        }
    };

    public static void buildAllStats(JsonObject json, ClientPlayerEntity player){
        buildBaseMinecraftStats(json, player);
        buildLifeStats(json, player);
        buildPositionStats(json, player);
        buildBiomeStats(json, player);
        buildWeatherStats(json, player);
    }


    /** Matches the mc_constants.json format for statistics ensuring all will be added to the provided json object.
     * @param json a JSON object into which the stats will be added (at the root, be careful of conflicts). Stats that are 0 will be omitted, loading a save from a different username will break most stats
     */
    public static void buildBaseMinecraftStats(JsonObject json, ClientPlayerEntity player)
    {
        if (Minecraft.getInstance().getConnection() == null)
            return;
        Minecraft.getInstance().getConnection().sendPacket(new CClientStatusPacket(CClientStatusPacket.State.REQUEST_STATS));
        StatisticsManager statisticsManager = player.getStats();

        for(Stat<?> stat : statisticsManager.getKeys()) {
            // Skip the "minecraft" namespace.
            String statTypeName = stat.getName().split(":")[0].substring("minecraft.".length());
            String statName = stat.getName().split(":")[1].substring("minecraft.".length());

            // First check if statTypeName is already in the statsJson object
            if(!json.has(statTypeName)) {
                json.add(statTypeName, new JsonObject());
            }
            // Then add the stat to the statTypeName object
            json.getAsJsonObject(statTypeName).addProperty(statName, statisticsManager.getValue(stat));

            // Debugging
            System.out.println( statTypeName + "." + statName);
        }
    }

    /** Builds the basic achievement world data to be used as observation signals by the listener.
     * @param json a JSON object into which the achievement stats will be added.
     */
    public static void oldBuildBaseMinecraftStats(JsonObject json, ClientPlayerEntity player)
    {
        Minecraft.getInstance().getConnection().sendPacket(new CClientStatusPacket(CClientStatusPacket.State.REQUEST_STATS));
        StatisticsManager statisticsManager = player.getStats();

//        json.addProperty("distance_travelled_cm",
//            statisticsManager.getValue(Stat(Stats.WALK_ONE_CM)
//            + statisticsManager.getValue(Stats.CROUCH_ONE_CM.getPath())
//            + statisticsManager.getValue(StatList.SPRINT_ONE_CM)
//            + statisticsManager.readStat(StatList.SWIM_ONE_CM)
//            + statisticsManager.readStat(StatList.FALL_ONE_CM)
//            + statisticsManager.readStat(StatList.CLIMB_ONE_CM)
//            + statisticsManager.readStat(StatList.FLY_ONE_CM)
//            + statisticsManager.readStat(StatList.DIVE_ONE_CM)
//            + statisticsManager.readStat(StatList.MINECART_ONE_CM)
//            + statisticsManager.readStat(StatList.BOAT_ONE_CM)
//            + statisticsManager.readStat(StatList.PIG_ONE_CM)
//            + statisticsManager.readStat(StatList.HORSE_ONE_CM)
//            + statisticsManager.readStat(StatList.AVIATE_ONE_CM)
//            );

        // TODO the logic below is for mineRL compat sake, may need to
        // revisit at some point
        for(Stat stat : statisticsManager.getKeys()) {
            // For MineRL, split over . and convert all camelCase to snake_case
            String[] stat_fields = stat.getName().split("\\.");
            JsonObject head = json;
            for (String unformatted_token : stat_fields) {
                if (unformatted_token.equals("minecraft")) {
                    continue;
                }
                String token = CaseFormat.UPPER_CAMEL.to(CaseFormat.LOWER_UNDERSCORE, unformatted_token);
                if (token.endsWith(":minecraft")) {
                    token = token.replace(":minecraft", "");
                }
                // Last element is a leaf
                if (unformatted_token.equals(stat_fields[stat_fields.length - 1])) {
                    // BAH map drop stat to items_dropped to prevent hash collision in dict keys
                    // MUST change this in CraftingHelper.java as well!!!! (search above comment)
                    if (token.equals("drop"))
                        token = "items_dropped";
                    head.addProperty(token, statisticsManager.getValue(stat));
                } else {
                    if (head.has(token))
                        if (head.get(token) instanceof JsonObject)
                            head = head.getAsJsonObject(token);
                        else {
                            System.out.println("Duplicate token! " + Arrays.toString(stat_fields));
                            head.remove(token);
                            JsonObject newRoot = new JsonObject();
                            head.add(token, newRoot);
                            head = newRoot;
                        }
                    else {
                        JsonObject newRoot = new JsonObject();
                        head.add(token, newRoot);
                        head = newRoot;
                    }
                }
            }
        }
    }

    /** Builds the basic life world data to be used as observation signals by the listener.
     * @param json a JSON object into which the life stats will be added.
     */
    public static void buildLifeStats(JsonObject json, ClientPlayerEntity player)
    {
        JsonObject lifeStats = new JsonObject();
        lifeStats.addProperty("life", player.getHealth());
        lifeStats.addProperty("score", player.getScore());    // Might always be the same as XP?
        lifeStats.addProperty("food", player.getFoodStats().getFoodLevel());
        lifeStats.addProperty("saturation", player.getFoodStats().getSaturationLevel());
        lifeStats.addProperty("xp", player.experienceTotal);
        lifeStats.addProperty("is_alive", player.isAlive());
        lifeStats.addProperty("air", player.getAir());
        lifeStats.addProperty("name", player.getName().toString());
        json.add("life_stats", lifeStats);
    }
    /** Builds the player position data to be used as observation signals by the listener.
     * @param json a JSON object into which the positional information will be added.
     */
    public static void buildPositionStats(JsonObject json, ClientPlayerEntity player)
    {
        json.addProperty("xpos",  player.getPosX());
        json.addProperty("ypos",  player.getPosY());
        json.addProperty("zpos", player.getPosZ());
        json.addProperty("pitch",  player.rotationPitch);
        json.addProperty("yaw", player.rotationYaw);
    }

    /** Builds the player's biome data.
     * @param json a JSON object into which the biome information will be added.
     * @param player - Non-null, must have player.world
     */
    public static void buildBiomeStats(JsonObject json, ClientPlayerEntity player)
    {
        BlockPos playerPos = player.getPosition();
        Biome playerBiome = player.world.getBiome(playerPos);
        // Name of the current biome
        json.addProperty("biome_name", playerBiome.toString());
        // ID of the current biome
        // json.addProperty("biome_id", RegistryKey.getOrCreateKey(Registry.BIOME_KEY, playerBiome.toString()).);
        // The average temperature of the current biome
        json.addProperty("biome_temperature", playerBiome.getTemperature());
        // The average rainfall chance of the current biome
        json.addProperty("biome_downfall", playerBiome.getDownfall());
        // The water level for oceans and rivers
        json.addProperty("sea_level", player.world.getSeaLevel());
    }

    /** Builds the player's weather information
     * @param json a JSON object into which the weather information will be added.
     * @param player - Non-null, must have player.world
     */
    public static void buildWeatherStats(JsonObject json, ClientPlayerEntity player)
    {
        BlockPos playerPos = player.getPosition();
        json.addProperty("light_level", player.world.getLight(playerPos));
        // If it is currently precipitating here
        json.addProperty("is_raining", player.world.isRaining());
        // If the playerPos has LOS to the sky
        json.addProperty("can_see_sky", player.world.canSeeSky(playerPos));
        // [0, 1] Brightness factor of the sun
        // json.addProperty("sun_brightness", player.world.getSunBrightnessFactor(0));
        // [0, 1] Light level provided by the sky
        // json.addProperty("sky_light_level", player.world.getSunBrightness(0));
        // TODO add other statuses such as is_raining or other current weather
    }

//    public static void buildEnvironmentStats(JsonObject json, ClientPlayerEntity player)
//    {
//        // json.addProperty("world_time", player.world.getWorldTime());  // Current time in ticks
//        // json.addProperty("total_time", player.world.getTotalWorldTime());  // Total time world has been running
//    }
//    /**
// * Build a signal for the cubic block grid centred on the player.<br>
// * Default is 3x3x4. (One cube all around the player.)<br>
// * Blocks are returned as a 1D array, in order
// * along the x, then z, then y axes.<br>
// * Data will be returned in an array called "Cells"
// * @param json a JSON object into which the info for the object under the mouse will be added.
// * @param environmentDimensions object which specifies the required dimensions of the grid to be returned.
// * @param jsonName name to use for identifying the returned JSON array.
// */
}
